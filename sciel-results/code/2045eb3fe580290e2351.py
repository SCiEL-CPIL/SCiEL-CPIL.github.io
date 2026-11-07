"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Katsuura function is multimodal with many local minima.
        # Strategy: CMA-ES with restarts based on stagnation.

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline

        # We'll use a simple CMA-ES implementation with restarts.
        # For 5D, CMA-ES works well. We do multiple restarts if stagnation is detected.

        max_restarts = 10
        calls_per_restart = max(30, budget // max_restarts)

        for restart in range(max_restarts):
            if calls_used >= budget:
                break
            if not time_ok():
                break

            # Initialize: first restart at origin, subsequent at random points
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            if not time_ok() or calls_used >= budget:
                break

            # CMA-ES parameters
            sigma = 1.0
            pop_size = 4 + int(3 * np.log(dim))  # for dim=5, pop_size=4+3*1.609=8 (int)
            pop_size = max(pop_size, dim + 2)

            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            step_size = 2.04 + np.log(dim)  # for dim=5, ~3.61

            # Weights
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Track best in population for stagnation detection
            local_best_f = fx
            no_improve_count = 0
            max_no_improve = 50

            while calls_used < budget and time_ok():
                # Generate population
                ys = []
                for i in range(pop_size):
                    if not time_ok() or calls_used >= budget:
                        break
                    y = x + sigma * (C_inv_sqrt @ rng.standard_normal(dim))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    ys.append((fy, y))

                if len(ys) < mu:
                    break

                ys.sort(key=lambda t: t[0])
                if ys[0][0] < local_best_f:
                    local_best_f = ys[0][0]
                    no_improve_count = 0
                    if ys[0][0] < best_f:
                        best_f, best_x = ys[0][0], ys[0][1].copy()
                else:
                    no_improve_count += 1

                if no_improve_count >= max_no_improve:
                    break

                # Update mean
                y_w = np.array([ys[i][1] for i in range(mu)])
                y_mean = np.sum(weights[:, None] * y_w, axis=0)
                x_old = x.copy()
                x = x + (y_mean - x)

                # Update sigma
                diff = (y_mean - x_old) / sigma
                if np.linalg.norm(diff) > 1e-16:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / np.linalg.norm(diff)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update C
                if np.linalg.norm(p_sigma) / chi_n > 1.4 + 2 / (dim + 1):
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / np.linalg.norm(diff)
                else:
                    p_c = (1 - c_c) * p_c

                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None, None] * np.outer(y_w - x_old, y_w - x_old), axis=0)
                C = (1 + c_1 + c_mu - c_c) * C + c_1 * (rank_one - C) + c_mu * rank_mu

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T

                # Compute C_inv_sqrt
                eigvals2, eigvecs2 = np.linalg.eigh(C)
                eigvals2 = np.clip(eigvals2, 1e-12, None)
                C_inv_sqrt = (eigvecs2 / np.sqrt(eigvals2)) @ eigvecs2.T

        return best_x, best_f
# EVOLVE-BLOCK-END
