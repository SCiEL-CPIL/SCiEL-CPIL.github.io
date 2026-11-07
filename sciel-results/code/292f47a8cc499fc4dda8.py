"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        # CMA-ES implementation
        def cma_es_run(x0, sigma0, budget_left, rng, func):
            nonlocal best_x, best_f, total_calls

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - cc + 2 * mueff / (n - 2)) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evals = 0
            best_local_x = x_mean.copy()
            best_local_f = safe_call(x_mean)
            if best_local_f is None:
                return best_local_x, np.inf
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()

            while evals < budget_left:
                # Check deadline
                if time.time() > deadline:
                    break
                # Check budget
                if total_calls >= budget:
                    break

                # Generate lambda offspring
                offspring = []
                fitness = []
                for i in range(lam):
                    if time.time() > deadline or total_calls >= budget:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    if fy is not None:
                        offspring.append((y, fy))
                        fitness.append(fy)
                        evals += 1
                        if fy < best_local_f:
                            best_local_f = fy
                            best_local_x = y.copy()
                            if fy < best_f:
                                best_f = fy
                                best_x = y.copy()

                if len(offspring) < 2:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                offspring = offspring[:mu]
                fitness = [t[1] for t in offspring]

                # Update mean
                new_mean = np.sum(np.array([w[i] * offspring[i][0] for i in range(mu)]), axis=0)

                # Update covariance matrix
                y_inv = [B.T @ (D * (offspring[i][0] - x_mean) / sigma) for i in range(mu)]
                delta = (new_mean - x_mean) / sigma
                pc = (1 - cc) * pc + cc * delta
                hsig = (np.sqrt((1 - (1 - cc) ** (evals + 1)) / (1 - (1 - cc) ** (evals + lam))) *
                        (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (evals + 1)) / chi_n < 1.4 + 2 / (n + 1)))
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * delta
                C = C + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(
                    [w[i] * np.outer(y_inv[i], y_inv[i]) for i in range(mu)], axis=0
                ) - (c1 + cmu) * C
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigenvalues, B = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                D = np.sqrt(eigenvalues)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                x_mean = new_mean
                x_mean = np.clip(x_mean, -5, 5)

            return best_local_x, best_local_f

        # Multi-start strategy
        rng = np.random.default_rng(42)

        # First try: center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        budget_per_run = max(50, budget // 4)

        # Run CMA-ES from center
        x_res, f_res = cma_es_run(x0, sigma0, budget_per_run, rng, func)

        # If we still have budget, try a few random starts
        if total_calls < budget and time.time() < deadline:
            for attempt in range(3):
                if total_calls >= budget or time.time() > deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                sigma_rand = 1.0
                remaining = budget - total_calls
                if remaining < 30:
                    break
                x_res2, f_res2 = cma_es_run(x_rand, sigma_rand, min(remaining, budget_per_run), rng, func)

        # Final fallback: if we barely used the budget, do a simple local search
        if total_calls < budget and time.time() < deadline and best_x is not None:
            x_cur = best_x.copy()
            f_cur = best_f
            sigma_local = 0.1
            for _ in range(min(200, budget - total_calls)):
                if time.time() > deadline or total_calls >= budget:
                    break
                y = np.clip(x_cur + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < f_cur:
                    x_cur, f_cur = y, fy
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = y.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.85

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
