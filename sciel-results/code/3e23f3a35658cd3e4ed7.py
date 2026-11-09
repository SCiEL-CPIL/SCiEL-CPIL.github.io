"""CMA-ES with adaptive restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        total_budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            return fx

        # Try multiple restarts of CMA-ES with different initializations
        num_restarts = 3
        budget_per_restart = total_budget // num_restarts

        for restart in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline - 5:
                break

            # Initial point: center for first, random for others
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            fx0 = call_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            if calls_made >= total_budget:
                break

            # CMA-ES parameters
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            sigma = 1.0  # initial step size
            x_mean = x0.copy()
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Adaptation parameters
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

            B = np.linalg.cholesky(C)
            D = np.ones(n)

            for gen in range(budget_per_restart):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline - 2:
                    break

                # Generate offspring
                ys = np.zeros((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    ys[i] = y

                # Evaluate
                fs = np.array([call_func(ys[i]) for i in range(lam)])
                if calls_made >= total_budget:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()

                # Weighted mean of best mu
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * ys_sorted[i]
                x_new = y_w

                # Update step size (CSA)
                y_sigma = (C_sqrt_inv @ x_mean)  # not quite right, let's simplify
                # Use simpler sigma update based on success
                # Count successes in this generation
                successes = np.sum(fs_sorted[:mu] < fs[order[mu]] if mu < lam else 0)
                if successes > 0:
                    sigma *= 1.0 + 0.1 * (successes / mu - 0.5)
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                x_mean = x_new

                # Update evolution paths
                v = B @ (D * (y_w - x_mean) / sigma) if sigma > 0 else np.zeros(n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.trace(np.eye(n)) else 0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * (x_mean - x_new) / sigma)) if sigma > 0 else (1 - c_c) * p_c

                # Rank-mu update
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                C += c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    zi = (ys_sorted[i] - x_mean) / sigma if sigma > 0 else np.zeros(n)
                    C += c_mu * weights[i] * np.outer(zi, zi)

                # Ensure positive definite
                try:
                    B = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                    D = np.sqrt(np.diag(C))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)

                # Check convergence
                if sigma < 1e-10:
                    break

        # Final local polish with (1+1)-ES
        if calls_made < total_budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.1
            for _ in range(total_budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
