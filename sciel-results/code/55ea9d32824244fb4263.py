"""CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, covariance adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        # CMA-ES parameters
        pop_size = 10
        # Initial step size
        sigma = 2.0
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Inverse square root of C for efficient sampling
        C_inv_sqrt = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Adaptation rates
        mu = pop_size // 2  # number of parents
        weights = np.ones(mu)
        weights[-1] = 2.0  # extra weight for the best
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] + weights[1] - 1) / ((dim + 2) ** 2 + mu_eff))

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Precompute Cholesky of C_inv_sqrt for sampling
        def update_chol(C_mat):
            """Compute inverse square root via eigendecomposition for stability."""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.maximum(eigvals, 1e-20)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        C_inv_sqrt = update_chol(C)

        while calls < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Sample population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            f_offspring = []
            for y in offspring:
                if calls >= budget:
                    break
                fy = func(y)
                f_offspring.append(fy)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(f_offspring) < pop_size:
                break

            # Sort by fitness
            indices = np.argsort(f_offspring)
            offspring_sorted = [offspring[i] for i in indices]
            f_sorted = [f_offspring[i] for i in indices]

            # Mean of selected parents
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring_sorted[i]

            # Update evolution paths
            diff = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            h_sig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - pop_size))) \
                if calls > pop_size else 1.0
            h = min(1.0, h_sig / (1 + 1e-15))

            p_c = (1 - c_c) * p_c + h * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                d = (offspring_sorted[i] - x) / sigma
                outer_sum += weights[i] * np.outer(d, d)

            delta_h = (1 - h) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim)) + c_mu * outer_sum

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update inverse square root
            C_inv_sqrt = update_chol(C)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Move center
            x = x_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
