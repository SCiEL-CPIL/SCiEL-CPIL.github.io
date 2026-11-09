"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_evals):
            nonlocal eval_count
            if eval_count >= max_evals:
                return

            n = self.dim
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            fx = safe_func(x)

            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Covariance matrix and its Cholesky decomposition
            C = np.eye(n)
            C_inv = np.eye(n)
            # Use eigendecomposition for step size adaptation
            B = np.eye(n)
            D = np.ones(n)

            # State variables
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            hsig = 0.0
            hsig_bar = 1.0 / damps

            # Precompute weights for ranking
            w = weights

            for gen in range(max_evals // lam + 1):
                if eval_count >= max_evals or time.time() > deadline:
                    break

                # Generate offspring
                offspring = []
                for i in range(lam):
                    # Transform standard normal using current B and D
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x_new = np.clip(x + sigma * y, -5, 5)
                    offspring.append((x_new, y, z))

                # Evaluate offspring
                for i, (x_new, _, _) in enumerate(offspring):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    offspring[i] = (x_new, offspring[i][1], offspring[i][2], safe_func(x_new))

                if eval_count >= max_evals or time.time() > deadline:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[3])

                # Update mean
                x_old = x.copy()
                for i in range(mu):
                    x = x + w[i] * offspring[i][0]

                # Update sigma (step size)
                y_mean = (x - x_old) / sigma if sigma > 0 else np.zeros(n)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean / chi_n
                p_sigma_norm = np.linalg.norm(p_sigma)
                hsig = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1) / damps)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
                sigma *= np.exp(cs / damps * (p_sigma_norm / chi_n - 1))

                # Update covariance matrix
                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean
                delta_hsig = (1 - hsig) * hsig_bar * (1 - hsig_bar)

                # Use rank-one and rank-mu updates
                C = (1 + c1 * delta_hsig + cmu - 1 - c1 - cmu) * C
                # Rank-one update
                C += c1 * (np.outer(p_c, p_c) + (1 - hsig) * delta_hsig * C)
                # Rank-mu update
                y_w = [offspring[i][1] for i in range(mu)]
                C += cmu * np.sum([w[i] * np.outer(y_w[i], y_w[i]) for i in range(mu)], axis=0)

                # Re-orthogonalize C and update B, D
                # Use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

                # Ensure sigma doesn't get too small or too large
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check convergence
                if np.all(np.abs(x - x_old) < 1e-10) and sigma < 1e-10:
                    break

            return x, best_f

        # Multi-start strategy
        # First, try a few random starts to find good basins
        num_starts = min(5, total_budget // 100)
        remaining_budget = total_budget

        for start_i in range(num_starts):
            if eval_count >= total_budget or time.time() > deadline:
                break

            # Random start
            x0 = rng.uniform(-5, 5, n)
            # Small initial sigma for local search
            sigma0 = 0.5

            # Give each start a portion of the budget
            if start_i < num_starts - 1:
                budget_for_start = min(remaining_budget - 50, 100)
            else:
                budget_for_start = remaining_budget

            if budget_for_start <= 0:
                break

            x, f = cma_es(x0, sigma0, budget_for_start)
            remaining_budget = total_budget - eval_count

        # Final refinement if budget remains
        if best_x is not None and eval_count < total_budget and time.time() < deadline:
            # Do a fine-grained local search around the best point
            x0 = best_x.copy()
            sigma0 = 0.1
            cma_es(x0, sigma0, total_budget - eval_count)

        return best_x, best_f
# EVOLVE-BLOCK-END
