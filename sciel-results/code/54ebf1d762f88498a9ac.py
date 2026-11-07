"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with proper covariance updates and multi-start for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def cma_es(x0, sigma0, remaining_budget, rng_local, time_deadline):
            """Standard CMA-ES implementation."""
            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            # CMA-ES parameters
            lambda_ = max(4 * n, 10)
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
            cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)) * (2 - cc))

            hsig = (1 - cc) / np.sqrt(1 - cc ** 2)

            best_x = x0.copy()
            best_f = func(x0)
            calls = 1

            # Precompute C^{-1/2} via eigendecomposition
            def compute_c_inv_sqrt(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.maximum(eigvals, 1e-15)
                c_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return c_inv_sqrt

            C_inv_sqrt = compute_c_inv_sqrt(C)

            gen = 0
            while calls < remaining_budget:
                if time.time() > time_deadline - 3:
                    break

                gen += 1
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_):
                    if calls >= remaining_budget:
                        break
                    if time.time() > time_deadline - 3:
                        break
                    z = rng_local.standard_normal(n)
                    y = x + sigma * C_inv_sqrt @ z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fs.append(fy)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(fs)
                ys_sorted = [ys[i] for i in sorted_idx[:mu]]
                fs_sorted = [fs[i] for i in sorted_idx[:mu]]

                # Update mean
                x_old = x.copy()
                x = sum(w * y for w, y in zip(weights, ys_sorted))
                x = np.clip(x, -5, 5)

                # Compute step vector
                d = x - x_old
                # Transform step by C^{-1/2}
                C_inv_sqrt_d = C_inv_sqrt @ d
                norm_d = np.linalg.norm(C_inv_sqrt_d)

                # Update sigma
                if norm_d > 1e-12:
                    p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_inv_sqrt_d / norm_d
                    sigma *= np.exp(cs / (n + 2) * (norm_d / np.sqrt(2) - 1))
                    sigma = np.clip(sigma, 1e-12, 100.0)

                # Update C: rank-1 update
                C = (1 + c1 * cc) * C + c1 * np.outer(d, d)

                # Update C: rank-mu update
                y_diffs = [y - x_old for y in ys_sorted]
                y_diffs_transformed = [C_inv_sqrt @ yd for yd in y_diffs]
                h = np.zeros(n)
                for w, ydt in zip(weights, y_diffs_transformed):
                    h += w * ydt
                C += cmu * np.outer(h, h)

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

                # Update p_c
                if norm_d > 1e-12:
                    p_c = (1 - cc) * p_c + cc * hsig * (norm_d / np.sqrt(2)) * C_inv_sqrt_d / (norm_d + 1e-12)

                # Update C with p_c (path covariance adaptation)
                C = (1 - c1 * cc * (2 - cc)) * C + c1 * (1 + cc * (2 - cc)) * np.outer(p_c, p_c)

                # Ensure C is positive definite again
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

                # Recompute C^{-1/2}
                C_inv_sqrt = compute_c_inv_sqrt(C)

            return best_x, best_f, calls

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Random initialization - sample a few points to find a good start
        n_random = min(30, budget // 10)
        for i in range(n_random):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10:
                break
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        # Phase 2: Run CMA-ES from the best starting point
        remaining = budget - total_calls
        if remaining > 20:
            x_opt, f_opt, calls_used = cma_es(best_x, 2.0, remaining, rng, deadline)
            total_calls += calls_used
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: If budget remains, do a second CMA-ES run with a smaller sigma
        if total_calls < budget and time.time() < deadline - 20:
            remaining = budget - total_calls
            if remaining > 20:
                # Start from the best found, with smaller sigma for local refinement
                x_opt, f_opt, calls_used = cma_es(best_x, 0.5, remaining, rng, deadline)
                total_calls += calls_used
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        # Phase 4: If still budget remains, try a third restart from a different random point
        if total_calls < budget and time.time() < deadline - 15:
            x_try = rng.uniform(-5, 5, n)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

            remaining = budget - total_calls
            if remaining > 20:
                x_opt, f_opt, calls_used = cma_es(best_x, 1.5, remaining, rng, deadline)
                total_calls += calls_used
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
