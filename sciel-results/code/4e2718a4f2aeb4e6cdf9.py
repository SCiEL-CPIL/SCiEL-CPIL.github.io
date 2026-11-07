"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Multiple restarts with different seeds and initial points
        restart_configs = [
            (0, "uniform_wide"),
            (1, "origin"),
            (2, "uniform_mid"),
            (3, "uniform_wide"),
            (4, "uniform_mid"),
            (5, "origin"),
            (6, "uniform_wide"),
            (7, "uniform_mid"),
        ]

        for seed, init_type in restart_configs:
            if total_calls >= budget:
                break
            if time.time() >= deadline - 15:
                break

            rng = np.random.default_rng(seed)

            # Initialize based on type
            if init_type == "uniform_wide":
                x = rng.uniform(-5, 5, dim)
            elif init_type == "origin":
                x = np.zeros(dim)
            elif init_type == "uniform_mid":
                x = rng.uniform(-3, 3, dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES parameters
            pop_size = min(12, max(6, budget // 40))
            mu = max(1, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)

            sigma = 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            restart_best_f = fx
            restart_best_x = x.copy()
            restart_start_f = fx

            gen = 0
            no_improve_count = 0
            threshold = 1e-10

            while total_calls < budget:
                if time.time() >= deadline - 10:
                    break

                batch_size = min(pop_size, budget - total_calls)
                if batch_size <= 0:
                    break

                # Generate offspring
                offspring = np.zeros((batch_size, dim))
                for i in range(batch_size):
                    z = rng.standard_normal(dim)
                    try:
                        L = np.linalg.cholesky(C)
                        step = L @ z
                    except np.linalg.LinAlgError:
                        step = z
                    offspring[i] = x + sigma * step
                    offspring[i] = np.clip(offspring[i], -5, 5)

                # Evaluate
                offspring_f = np.empty(batch_size)
                for i in range(batch_size):
                    offspring_f[i] = func(offspring[i])
                    total_calls += 1
                    if offspring_f[i] < restart_best_f:
                        restart_best_f = offspring_f[i]
                        restart_best_x = offspring[i].copy()
                    if offspring_f[i] < best_f:
                        best_f = offspring_f[i]
                        best_x = offspring[i].copy()

                if total_calls >= budget:
                    break

                # Sort and select
                order = np.argsort(offspring_f)
                selected = offspring[order[:mu]]
                m_new = (weights[:, None] * selected).sum(axis=0)

                # Update step size
                try:
                    C_chol = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_chol)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)

                diff = m_new - x
                p_sigma = (1 - 1 / (dim + 4)) * p_sigma + (1 / (dim + 4)) * np.sqrt(mu_eff) * (C_inv_sqrt @ diff)
                norm_ps = np.linalg.norm(p_sigma)
                hat_chi = np.sqrt(mu_eff) * (1 - 1 / (2 * (dim + 2)))
                sigma *= np.exp((1 / (dim + 4 + 2)) * (norm_ps / hat_chi - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 20.0)

                # Update mean
                x = m_new

                # Update covariance
                p_c = (1 - 1 / (dim + 4)) * p_c + (1 / (dim + 4)) * np.sqrt(mu_eff) * diff
                # Rank-mu update
                try:
                    C_inv = np.linalg.inv(C)
                    outer = np.outer(p_c, p_c)
                    C = (1 + 2 / (dim + 4 + 2) * (1 / (dim + 4 + 2) - 1) * (1 - 1 / (dim + 4))) * C \
                        + (1 + 2 / (dim + 4 + 2) * (1 / (dim + 4 + 2) - 1) * (1 - 1 / (dim + 4))) * outer \
                        + (1 - 1 / (dim + 4)) * C
                except np.linalg.LinAlgError:
                    pass

                # Simpler covariance update: just use the step size adaptation
                # and rely on restarts for diversification
                # Let's use a simpler approach: update C based on selected points
                if gen > 0:
                    # Compute covariance of selected points
                    centered = selected - m_new
                    C_new = (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
                    # Mix with old C
                    C = 0.9 * C + 0.1 * C_new
                    # Ensure positive semi-definite
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-8)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)

                gen += 1

                # Check for stagnation
                if restart_best_f - restart_start_f < threshold * max(1, abs(restart_start_f)):
                    no_improve_count += 1
                else:
                    no_improve_count = 0

                if no_improve_count > 50:
                    break

            # End of this restart
            if restart_best_f < best_f:
                best_f = restart_best_f
                best_x = restart_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
