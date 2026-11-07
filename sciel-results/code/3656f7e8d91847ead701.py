"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box optimization on BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        # Track best solution found
        best_x = None
        best_f = np.inf
        calls_used = 0

        # CMA-ES parameters
        # For dim=5, lambda is typically 4+floor(3*ln(dim)) = 4+3*1.609 = 8
        lambda_ = max(4 + int(3 * np.log(dim)), 8)
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        def cma_es(x_center, sigma, max_calls):
            nonlocal best_x, best_f, calls_used

            x = x_center.copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Generate population
                W = B @ D
                # Sample lambda_ points
                ys = rng.standard_normal((lambda_, dim))
                xs = x + sigma * (ys @ W.T)
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.array([func(xi) for xi in xs])
                calls_used += lambda_

                # Update best
                for i in range(lambda_):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xs[i].copy()

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]

                # Update mean
                x_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update step size (CSA)
                y_mean = (x_new - x) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < 1.4 + 2 / (dim + 1)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Update covariance matrix
                pc = (1 - c_c) * pc + hsig * c_c * y_mean
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                diff = np.zeros((dim, dim))
                for i in range(mu):
                    yi = (xs_sorted[i] - x) / sigma
                    diff += weights[i] * np.outer(yi, yi)
                C = (1 + c_1 * delta_hsig - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * diff

                # Eigendecomposition for C = B @ diag(D^2) @ B.T
                D2, B = np.linalg.eigh(C)
                D2 = np.maximum(D2, 1e-12)
                D = np.sqrt(D2)
                x = x_new

                # Check convergence
                if sigma < 1e-15 or np.all(np.abs(x - x_center) < 1e-10):
                    break

            return best_x, best_f

        # Multi-start CMA-ES with restarts
        # First: try a few random starts with small budgets to find good basins
        n_initial = min(10, budget // 50)
        for i in range(n_initial):
            if time.time() > deadline or calls_used >= budget:
                break
            x0 = rng.uniform(-4, 4, dim)
            # Small CMA-ES run
            max_calls_for_this = max(20, budget // (n_initial + 1))
            cma_es(x0, 1.5, max_calls_for_this)

        # Then: run longer CMA-ES from the best point found
        remaining = budget - calls_used
        if remaining > 20 and best_x is not None:
            cma_es(best_x.copy(), 1.0, remaining)

        # Final polish: if we have budget left, do fine-tuning around best
        remaining = budget - calls_used
        if remaining > 10 and best_x is not None:
            # Local search with smaller sigma
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if time.time() > deadline or calls_used >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
