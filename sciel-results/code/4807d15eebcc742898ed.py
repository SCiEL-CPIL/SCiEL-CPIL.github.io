"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts and time guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # 320s safety margin within 330s kill

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        def cma_es(func, x0, sigma0, budget, dim, rng, deadline):
            n = dim
            # Population size
            lambda_pop = int(4 + 3 * np.log(n))
            # Weights
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, lambda_pop + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            # Step size control
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # Covariance matrix adaptation
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Chi-squared expected value
            chi_n = n * (1 - 0.5 / (4 * n - 1)) + np.sqrt(max(0, n - 1)) * (
                1 - 1 / (31 * n ** 2)
            )

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            def mat_sqrt_inv(M):
                """Compute matrix square root and inverse square root."""
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.maximum(eigvals, 1e-20)
                sqrt_vals = np.sqrt(eigvals)
                inv_sqrt_vals = 1.0 / sqrt_vals
                M_sqrt = eigvecs @ np.diag(sqrt_vals) @ eigvecs.T
                M_inv_sqrt = eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T
                return M_sqrt, M_inv_sqrt

            C_sqrt, C_inv_sqrt = mat_sqrt_inv(C)

            best_x = x.copy()
            best_f = func(x)

            for it in range(budget):
                if time.time() > deadline:
                    break

                # Generate population
                offsets = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    offsets.append(sigma * C_sqrt @ z)

                # Evaluate
                ys = []
                fvals = []
                for i in range(lambda_pop):
                    y = np.clip(x + offsets[i], -5, 5)
                    fy = func(y)
                    ys.append(y)
                    fvals.append(fy)

                fvals = np.array(fvals)
                ys = np.array(ys)

                # Sort
                order = np.argsort(fvals)
                ys = ys[order]
                fvals = fvals[order]

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = ys[0].copy()

                # Update mean
                y_w = np.sum(weights[:mu, None] * ys[:mu], axis=0)
                x = np.clip(x + (y_w - x), -5, 5)

                # Update step size
                z_w = np.sum(weights[:mu, None] * (C_inv_sqrt @ (ys[:mu] - x_old).T).T, axis=0)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ z_w
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update evolution paths
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (y_w - x_old)

                # Update covariance
                delta = (1 + (1 - c_c) * c_c * (1 - c_c) * (np.dot(pc, C_inv_sqrt @ pc) > (1 + 4 / n) * (1 - c_c)) * np.outer(pc, pc)) - 1
                C = (1 + c_1 + c_mu - c_c) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(
                    weights[:mu, None, None] * np.outer((ys[:mu] - x_old).T, (ys[:mu] - x_old)), axis=0
                )

                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-15)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                C_sqrt, C_inv_sqrt = mat_sqrt_inv(C)

                x_old = x.copy()

                # Termination check
                if np.all(np.abs(x - x_old) < 1e-10) and sigma < 1e-10:
                    break

            return best_x, best_f

        # Multi-start strategy with different initializations
        rng = np.random.default_rng(42)

        # First try: center
        x0 = np.zeros(dim)
        sigma0 = 2.0
        b1 = budget // 2
        bx, bf = cma_es(func, x0, sigma0, b1, dim, rng, deadline)
        global_best_x, global_best_f = bx, bf

        if time.time() > deadline:
            return global_best_x, global_best_f

        # Second try: random start
        remaining = budget - b1
        x0 = rng.uniform(-4, 4, dim)
        sigma0 = 1.5
        bx, bf = cma_es(func, x0, sigma0, remaining, dim, rng, deadline)
        if bf < global_best_f:
            global_best_x, global_best_f = bx, bf

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
