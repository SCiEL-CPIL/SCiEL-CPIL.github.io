"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 320s hard limit
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = np.inf

        # Use a fixed seed for reproducibility within a run
        rng = np.random.default_rng(42)

        def cma_es(x0, sigma0, max_evals, rng_local, deadline_check):
            """Run CMA-ES from x0 for up to max_evals evaluations."""
            n = dim
            x = x0.copy()
            sigma = sigma0
            f_x = func(x)
            evals = 1

            # CMA-ES parameters
            lambda_pop = 4 + int(3 * np.log(n))
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            # Precompute sqrtm of C (identity initially)
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 0))
            B = eigvecs

            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            for _ in range(max_evals - 1):
                if deadline_check():
                    break

                # Sample offspring
                ys = []
                fs = []
                for i in range(lambda_pop):
                    z = rng_local.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    evals += 1
                    if deadline_check():
                        break

                if evals > max_evals:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order[:mu]]
                fs_sorted = [fs[i] for i in order[:mu]]

                # Update mean
                x_new = weights @ np.array(ys_sorted)

                # Update step size control variable
                y_mean = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update evolution path
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

                # Update covariance matrix
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 + delta_hsig) * C
                C -= c_1 * (np.outer(p_c, p_c) + (1 - hsig) * (2 - c_c) * C)
                C += c_mu * sum(wi * np.outer(y, y) for wi, y in zip(weights, (np.array(ys_sorted) - x) / sigma))

                # Symmetrize and ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 0)
                D = np.sqrt(eigvals)
                B = eigvecs

                x = x_new
                f_x = min(fs_sorted)

                # Check for stagnation
                if np.allclose(eigvals, 0, atol=1e-15):
                    break

            return x, f_x, evals

        # Multi-start strategy
        n_starts = 6
        evals_per_start = budget // n_starts

        for s in range(n_starts):
            if time.time() > deadline:
                break

            # Different starting points for each restart
            if s == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
            else:
                # Start near current best with some randomness
                x0 = best_x + rng.standard_normal(dim) * 1.0
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            local_rng = np.random.default_rng(42 + s)
            x_res, f_res, evals_used = cma_es(
                x0, sigma0, evals_per_start + (budget - evals_per_start * n_starts) if s == n_starts - 1 else evals_per_start,
                local_rng,
                lambda: time.time() > deadline
            )

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Final polishing with a small (1+1)-ES around the best point
        remaining = budget - sum(evals_per_start for _ in range(n_starts))
        if remaining > 0 and time.time() < deadline:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
