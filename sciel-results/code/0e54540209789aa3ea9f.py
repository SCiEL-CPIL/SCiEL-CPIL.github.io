"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive population and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        dim = self.dim
        budget = self.budget

        # Initial best
        best_x = np.zeros(dim)
        best_f = float('inf')

        # We'll use a simple CMA-ES-like strategy with restarts
        # Strategy: multiple short CMA-ES runs with different starting points

        def cma_run(x0, f0, max_evals, rng):
            """Run CMA-ES from x0 with f0, using at most max_evals evaluations."""
            n = dim
            x = x0.copy()
            f_best = f0
            x_best = x0.copy()
            sigma = 1.0
            lambda_ = max(4 + int(3 * np.log(n)), 6)  # population size
            mu = lambda_ // 2  # number of parents
            # weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            evals_used = 0

            for gen in range(max_evals // lambda_ + 1):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break

                # Generate lambda offspring
                offspring = []
                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    offspring.append((fy, y))

                if len(offspring) == 0:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < f_best:
                    f_best = offspring[0][0]
                    x_best = offspring[0][1].copy()

                # Take mu best
                if len(offspring) < mu:
                    break
                y_w = np.array([offspring[i][1] for i in range(mu)])
                f_w = np.array([offspring[i][0] for i in range(mu)])

                # Weighted mean
                m = (weights * y_w.T).T

                # Success rule for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * ((m - x) / sigma).T)).T
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_used // lambda_ + 1))) / chi_n < 1.4 + 2 / (n + 1)
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update C
                if hsig:
                    pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                diff = pc / np.sqrt(c_c * (2 - c_c) * mu_eff)
                c_cov = c_1 * (np.outer(diff, diff) + (1 - c_c) * (2 - c_c) * np.eye(n)) + c_mu * np.sum(weights[:, None] * np.outer((y_w - x).T / sigma, (y_w - x).T / sigma), axis=0)
                C = (1 - c_cov) * C + c_cov * (np.outer(diff, diff) + (1 - c_c) * (2 - c_c) * np.eye(n)) + c_mu * np.sum(weights[:, None] * np.outer((y_w - x).T / sigma, (y_w - x).T / sigma), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Eigenvalue decomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D, 1e-20, None))

                # Update mean
                x = m
                x = np.clip(x, -5, 5)

            return x_best, f_best, evals_used

        # Multi-restart strategy
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with large sigma
        # Try a few random starts with larger sigma
        num_starts = 5
        evals_per_start = max(20, budget // (num_starts * 3))

        for s in range(num_starts):
            if time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Run a short CMA-ES
            xb, fb, used = cma_run(x0, f0, evals_per_start, rng)
            if fb < best_f:
                best_f = fb
                best_x = xb.copy()

        # Phase 2: Local refinement around best point
        remaining = budget - 1 - num_starts - (num_starts - 1) * 0  # rough estimate
        # Actually let's just do more CMA-ES runs from best
        num_refine = 3
        evals_refine = max(30, (budget - 50) // num_refine)

        for r in range(num_refine):
            if time.time() > deadline:
                break
            # Perturb best slightly
            x0 = best_x + 0.5 * rng.standard_normal(dim)
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            xb, fb, used = cma_run(x0, f0, evals_refine, rng)
            if fb < best_f:
                best_f = fb
                best_x = xb.copy()

        # Phase 3: Fine-tune around best with small sigma
        if time.time() < deadline:
            x0 = best_x.copy()
            f0 = best_f
            # Simple local search with very small steps
            sigma_small = 0.05
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = np.clip(x0 + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < f0:
                    x0, f0 = y, fy
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                else:
                    sigma_small *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
