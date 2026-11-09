"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal CMA-ES / Natural Gradient Descent optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Population parameters
        lambda_ = 20
        mu_ = 10

        # Weights
        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial exploration
        init_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
        ]
        for i in range(dim):
            for val in [2.0, -2.0]:
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)

        for pt in init_points:
            if call_count >= budget - lambda_:
                break
            if time.time() > deadline:
                break
            _evaluate(pt)

        # Set initial mean to the best point found
        if best_x is not None:
            x_mean = best_x.copy()
        else:
            x_mean = np.zeros(dim)

        # Diagonal step-size vector (instead of full covariance)
        sigma_vec = np.full(dim, 2.0)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Chi_n constant for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Diagonal adaptation: track per-dimension step sizes
        # We use a rank-1 + rank-mu update on the diagonal

        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                y = x_mean + sigma_vec * z
                f_y = _evaluate(y)
                offspring.append((y, f_y, z))

            if len(offspring) < mu_:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu_]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)

            # Update evolution paths
            delta_mean = x_mean - old_mean

            # Normalize delta_mean by current step sizes for p_c update
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (delta_mean / sigma_vec)

            # Update sigma (global)
            z_mean = np.sum([w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z_mean

            norm_p_sigma = np.linalg.norm(p_sigma)
            global_sigma = np.prod(sigma_vec) ** (1.0 / dim)  # geometric mean
            global_sigma = global_sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            global_sigma = np.clip(global_sigma, 1e-10, 10.0)

            # Diagonal covariance adaptation: Rank-1 + Rank-mu on diagonal
            # The diagonal of the full covariance update is:
            # C_ii = (1 - c_1 - c_mu) * C_ii + c_1 * (p_c_i^2 + (1-(1-c_sigma)^2)) 
            #        + c_mu * sum_j w_j * (y_ji - mean_i)^2
            # Since we track sigma_vec, C_ii = sigma_vec_i^2

            # Compute per-dimension updates
            diag_C = sigma_vec ** 2

            # Rank-1 contribution to diagonal
            rank1_diag = p_c ** 2 + (1 - (1 - c_sigma) ** 2)

            # Rank-mu contribution to diagonal
            rank_mu_diag = np.zeros(dim)
            for i in range(mu_):
                diff = (selected[i][0] - x_mean) / sigma_vec  # normalize
                rank_mu_diag += w[i] * diff ** 2

            # Update diagonal
            diag_C_new = (1 - c_1 - c_mu) * diag_C + c_1 * rank1_diag + c_mu * rank_mu_diag

            # Ensure positivity and apply the global sigma scaling
            diag_C_new = np.maximum(diag_C_new, 1e-20)
            sigma_vec_new = np.sqrt(diag_C_new)

            # Scale by global sigma ratio to maintain overall scale
            ratio = global_sigma / (np.prod(sigma_vec) ** (1.0 / dim))
            sigma_vec_new = sigma_vec_new * ratio

            sigma_vec_new = np.clip(sigma_vec_new, 1e-10, 10.0)
            sigma_vec = sigma_vec_new

        return best_x, best_f
# EVOLVE-BLOCK-END
