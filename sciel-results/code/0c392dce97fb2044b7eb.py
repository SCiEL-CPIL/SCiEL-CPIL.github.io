"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for Sphere function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            nonlocal calls_made
            return calls_made >= budget or (time.time() - t0) > deadline

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_evals):
            nonlocal calls_made, best_f, best_x
            if check_time():
                return

            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Weights
            mu = lam // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            B = np.eye(n)
            D = 1.0
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            evals = 0
            while evals < max_evals and not check_time():
                # Generate offspring
                offspring = []
                for j in range(lam):
                    if check_time():
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    fy = evaluate(y)
                    offspring.append((fy, y, z))
                    evals += 1

                if not offspring:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                x_old = x.copy()
                x = x + y_w

                # Update path evolution strategies
                pc = (1 - c_c) * pc + c_c * (x - x_old) / (sigma * D) if sigma * D > 1e-15 else (1 - c_c) * pc
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (evals // lam + 1)))) / chi_n < (1.4 + 2 / (n + 1))
                ps = (1 - c_sigma) * ps + float(hsig) * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x - x_old) / (sigma * B @ (D * np.ones(n))) if sigma > 1e-15 else (1 - c_sigma) * ps

                # Update covariance matrix
                delta_hsig = 1 - float(hsig)
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = offspring[i][2]
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c_c * (2 - c_c - c_1) * delta_hsig * np.outer(pc, pc) - c_1 * C)
                C += (c_1 + c_mu) * (rank_mu - np.eye(n))

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    # Update eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    B = eigvecs
                    D = np.sqrt(eigvals)
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                    D = 1.0

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Check for stagnation
                if sigma < 1e-10:
                    break

            return

        # Phase 1: Quick random exploration to get a good starting point
        n_init = min(20, budget // 10)
        for i in range(n_init):
            if check_time():
                break
            x_init = rng.uniform(-5, 5, dim)
            evaluate(x_init)

        # Phase 2: CMA-ES from the best point found
        if best_x is not None:
            cma_es(best_x.copy(), 1.0, budget - calls_made)

        # Phase 3: If we still have budget, do a local refinement with smaller steps
        if not check_time() and best_x is not None:
            # Try a few random restarts with CMA-ES
            remaining = budget - calls_made
            n_restarts = min(3, remaining // 50)
            for r in range(n_restarts):
                if check_time():
                    break
                x_start = best_x + rng.standard_normal(dim) * 0.5
                x_start = np.clip(x_start, -5, 5)
                fx_start = evaluate(x_start)
                if fx_start < best_f:
                    best_f = fx_start
                    best_x = x_start.copy()
                cma_es(x_start, 0.3, remaining - (calls_made - (budget - remaining)))

        return best_x, best_f
# EVOLVE-BLOCK-END
