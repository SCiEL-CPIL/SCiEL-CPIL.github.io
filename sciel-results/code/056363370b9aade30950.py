"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard and global best tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.zeros(dim)
        best_fx = func(best_x)

        # CMA-ES parameters
        pop_size = 10  # λ
        # Initial sigma: explore the box
        sigma = 2.0

        # CMA-ES state
        mean = np.zeros(dim)
        C = np.eye(dim)
        # Precompute eigendecomposition for sampling
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)

        # CMA-ES weights
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        calls_made = 1  # initial best_x call

        def time_check():
            return time.time() < deadline

        # Main loop
        while calls_made < budget and time_check():
            # Generate population
            y_list = []
            x_list = []
            for i in range(pop_size):
                z = np.random.standard_normal(dim)
                y = B @ (D * z)
                x = np.clip(mean + sigma * y, -5, 5)
                y_list.append(y)
                x_list.append(x)

            # Evaluate population
            f_list = []
            for x in x_list:
                if not time_check():
                    break
                f = func(x)
                calls_made += 1
                f_list.append(f)
                if f < best_fx:
                    best_fx = f
                    best_x = x.copy()

            # If we ran out of time, break
            if not time_check():
                break

            # Sort by fitness
            idx = np.argsort(f_list)
            y_list = [y_list[i] for i in idx]
            x_list = [x_list[i] for i in idx]
            f_list = [f_list[i] for i in idx]

            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * x_list[i] for i in range(mu)], axis=0)

            # Update covariance matrix
            y_diff = np.array([y_list[i] for i in range(mu)])
            y_w = y_diff - np.outer(weights, np.ones(mu_eff))  # weighted mean
            # Actually, weighted mean of y
            y_w_mean = np.sum([weights[i] * y_list[i] for i in range(mu)], axis=0)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_N * (B @ (D * y_w_mean))

            # Hessian matrix for covariance update
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += weights[i] * np.outer(y_list[i] - y_w_mean, y_list[i] - y_w_mean)

            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - 1))) <
                       (1.4 + 2 / (dim + 1)) * chi_N)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            # Update C
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + \
                c_1 * np.outer(p_c + c_c * p_c - p_c, p_c + c_c * p_c - p_c) + \
                c_mu * H

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for stagnation: if best_fx hasn't improved in last 30 calls, restart
            # We track this by checking if mean has moved significantly
            # For simplicity, we'll rely on CMA-ES adaptation

            # If sigma is very small and we're not improving, we could restart
            # but let's trust CMA-ES for now

        return best_x, best_fx
# EVOLVE-BLOCK-END
