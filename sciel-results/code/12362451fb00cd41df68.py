"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for multimodal optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 330s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma, n_evals, max_stagnation=20):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            n = dim
            lam = max(6, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            xmean = x0.copy()
            C = np.eye(n)
            # Compute initial Cinv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-12))
            B = eigvecs
            Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            evals_used = 0
            no_improve = 0
            local_best = best_f
            sigma_min = 1e-10
            sigma_max = 10.0

            for gen in range(n_evals // lam + 10):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Check if sigma is too small or too large
                if sigma < sigma_min or sigma > sigma_max:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = xmean + sigma * Cinv_sqrt @ rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    ys.append((y, fy))

                if len(ys) < 2:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[1])

                # Check for improvement
                if ys[0][1] < local_best - 1e-12:
                    local_best = ys[0][1]
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart if stagnating
                if no_improve > max_stagnation:
                    break

                # Update mean
                xnew = np.zeros(n)
                for i in range(mu):
                    xnew += weights[i] * ys[i][0]

                # Update step size control variable
                old_xmean = xmean.copy()
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * Cinv_sqrt @ (xnew - old_xmean)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_used + 1) / n))
                         < (1.4 + 2 / (n + 1)) * chi_n)

                # Update evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * Cinv_sqrt @ (xnew - old_xmean)

                xmean = xnew

                # Update covariance matrix
                delta_hsig = (2 - cc) if hsig else 0.0
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = Cinv_sqrt @ (ys[i][0] - xmean)
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                # Ensure PSD
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2

                # Update D and B
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                Cinv_sqrt = B @ np.diag(1.0 / D) @ B.T

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, sigma_min)
                sigma = min(sigma, sigma_max)

                evals_used += len(ys)

                if evals_used >= n_evals:
                    break

        # Strategy: Multiple CMA-ES runs with different initializations
        # Phase 1: One run from center with large sigma
        if calls_made < budget and time.time() < deadline:
            x0 = np.zeros(dim)
            cma_es(x0, 3.0, 150, max_stagnation=25)

        # Phase 2: Run from best point with medium sigma
        if calls_made < budget and best_x is not None and time.time() < deadline:
            cma_es(best_x.copy(), 1.5, 120, max_stagnation=20)

        # Phase 3: Run from best point with smaller sigma
        if calls_made < budget and best_x is not None and time.time() < deadline:
            cma_es(best_x.copy(), 0.8, 100, max_stagnation=15)

        # Phase 4: Final refinement with very small sigma
        if calls_made < budget and best_x is not None and time.time() < deadline:
            cma_es(best_x.copy(), 0.3, budget - calls_made, max_stagnation=10)

        # Phase 5: If budget remains, do (1+1)-ES refinement
        if calls_made < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
