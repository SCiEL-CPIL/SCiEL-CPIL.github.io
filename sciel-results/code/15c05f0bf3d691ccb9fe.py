"""Combined CMA-ES with ASA refinement for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer followed by ASA local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        bound = self.bound

        # Global tracking
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -bound, bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: CMA-ES with ~55% of budget ---
        def cma_es(x0, sigma0, budget_left):
            nonlocal calls, best_x, best_f, deadline

            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            cs = (mueff + 2) / (n + mueff + 5)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)

            # Initial population
            rng = np.random.default_rng(42)
            pop = np.array([mean + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lam)])
            pop_f = np.array([eval_func(x) for x in pop])

            it = 0
            max_iter = 500
            while calls < budget_left and it < max_iter:
                if time.time() > deadline - 2.0:
                    break
                it += 1

                # Sort
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update mean
                old_mean = mean.copy()
                mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Step size control
                dx = (mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * (B.T @ dx)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B.T @ (D * (B.T @ dx)))

                # Covariance matrix update
                delta_d = cc * hsig + (1 - cc)
                delta_d = min(delta_d, 1.0)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean)

                C = (1 + c1 * delta_d - c1 - cmu) * C + c1 * rank_one + cmu / sigma**2 * rank_mu
                C = (C + C.T) / 2

                # Re-orthonormalize
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(n)
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 1e3)

                # Generate new population
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = mean + sigma * (B @ (D * z))
                    pop_f[i] = eval_func(pop[i])

                # Convergence check
                if np.linalg.norm(mean - old_mean) / max(sigma, 1e-15) < 1e-10:
                    break

            return mean, sigma

        # --- Phase 2: ASA refinement with remaining budget ---
        def asa(x0, budget_left):
            nonlocal calls, best_x, best_f, deadline

            n = dim
            x = x0.copy()
            f = best_f
            temp = 1.0
            cooling = 0.95
            max_iter = budget_left
            rng = np.random.default_rng(123)

            for it in range(max_iter):
                if calls >= budget_left or time.time() > deadline - 1.0:
                    break

                # Generate candidate
                delta = temp * rng.standard_normal(n)
                x_new = np.clip(x + delta, -bound, bound)
                f_new = eval_func(x_new)

                if f_new < f:
                    x = x_new
                    f = f_new
                else:
                    # Accept with probability
                    if rng.random() < np.exp(-(f_new - f) / max(temp, 1e-10)):
                        x = x_new
                        f = f_new

                temp *= cooling
                if temp < 1e-8:
                    break

            return x, f

        # --- Main execution ---
        # Initial evaluations to get a decent starting point
        rng_init = np.random.default_rng(0)
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng_init.uniform(-bound, bound, dim))

        for p in init_points:
            eval_func(p)

        # CMA-ES phase: use 55% of budget
        cma_budget = int(budget * 0.55)
        x_cma, sigma_cma = cma_es(best_x.copy(), 2.0, cma_budget)

        # ASA phase: use remaining budget
        asa_budget = budget - calls
        if asa_budget > 0 and time.time() < deadline - 1.0:
            x_asa, f_asa = asa(best_x.copy(), asa_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
