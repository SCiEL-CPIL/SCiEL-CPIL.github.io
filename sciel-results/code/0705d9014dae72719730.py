"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285
        rng = np.random.default_rng(7)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        def cma_es(x0, sigma0, budget_remaining, max_time):
            nonlocal best_x, best_f, calls_used

            # Population size
            lam = max(4, int(4 + 3 * np.log(d)))
            mu = lam // 2  # number of selected parents

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Step size control parameters
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

            # Initialization
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            C_inv = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            # Precompute sqrt(C) using Cholesky (we'll update it)
            def cholesky_safe(C):
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(d))
                    return L
                except:
                    # Fallback: eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    return L

            L = cholesky_safe(C)
            D = np.diag(np.sqrt(np.diag(C)))
            B = L @ np.diag(1.0 / np.sqrt(np.diag(C)))

            # Track best
            local_best_x = x_mean.copy()
            local_best_f = float('inf')

            # Generate and evaluate initial population
            while True:
                if calls_used >= max_calls or time.time() > max_time:
                    break

                # Generate population
                ys = np.zeros((lam, d))
                xs = np.zeros((lam, d))
                fs = np.full(lam, float('inf'))

                for i in range(lam):
                    if calls_used >= max_calls or time.time() > max_time:
                        break
                    yi = rng.standard_normal(d)
                    ys[i] = yi
                    xi = x_mean + sigma * (B @ (D @ yi))
                    xi = np.clip(xi, -5, 5)
                    fx = try_eval(xi)
                    if fx is None:
                        break
                    fs[i] = fx
                    xs[i] = xi

                if calls_used >= max_calls or time.time() > max_time:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                fs_sorted = fs[order]
                xs_sorted = xs[order]
                ys_sorted = ys[order]

                # Update best
                if fs_sorted[0] < local_best_f:
                    local_best_f = fs_sorted[0]
                    local_best_x = xs_sorted[0].copy()

                # Update mean
                x_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)

                # Update step size
                xbar_old = x_mean
                xbar_new = x_new
                D_inv = np.diag(1.0 / np.sqrt(np.diag(C)))
                B_inv = np.linalg.inv(B) if d <= 10 else B.T  # approximate
                # Actually use proper inverse
                try:
                    B_inv = np.linalg.inv(B)
                except:
                    B_inv = B.T

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                     (B_inv @ (D_inv @ (x_new - x_mean) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n < 1.4 + 2 / (d + 1))
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((d, d))
                for k in range(mu):
                    yk = B_inv @ (D_inv @ (xs_sorted[k] - x_mean) / sigma)
                    rank_mu += w[k] * np.outer(yk, yk)
                C = (1 + c_c * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                x_mean = x_new
                x_mean = np.clip(x_mean, -5, 5)

                # Update B and D
                L = cholesky_safe(C)
                D = np.diag(np.sqrt(np.diag(C)))
                try:
                    B = L @ np.diag(1.0 / np.sqrt(np.diag(C)))
                except:
                    B = L

                # Check for stagnation
                if local_best_f == best_f and np.linalg.norm(x_new - x_mean) < 1e-10:
                    break

        # Initial phase: try several starting points with CMA-ES
        start_points = [
            np.zeros(d),
            rng.uniform(-3, 3, d),
            rng.uniform(-5, 5, d),
        ]

        # Allocate budget: spend some on initial exploration, then CMA-ES
        # Try first a few random points to get a rough idea
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            fx = try_eval(sp)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()

        # Run CMA-ES from the best point found
        remaining = max_calls - calls_used
        if remaining > 10:
            cma_es(best_x.copy(), 1.0, remaining, deadline - 5)

        # If we have budget left, try another CMA-ES from a different start
        remaining = max_calls - calls_used
        if remaining > 20 and time.time() < deadline - 10:
            # Try from a random point
            x0 = rng.uniform(-4, 4, d)
            cma_es(x0, 2.0, remaining, deadline - 3)

        return best_x, best_f
