"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = int(4 + 3 * np.log(dim))  # lambda = 4 + 3*ln(n)
        mu = pop_size // 2  # number of selected individuals

        # Initial point and sigma
        x0 = np.zeros(dim)
        sigma = 2.0  # initial step size

        # CMA-ES state
        C = np.eye(dim)  # covariance matrix
        C_sqrt = np.eye(dim)  # square root of C
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)  # evolution path for C
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial population
        best_x = x0.copy()
        best_f = func(best_x)
        calls_made = 1

        for iteration in range(budget // pop_size):
            # Check deadline
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            pop = []
            fvals = []
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                z = np.random.standard_normal(dim)
                # Transform using C_sqrt
                y = x0 + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                pop.append(y)
                fvals.append(fy)

            if not pop:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = [pop[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Mean update
            mean = np.zeros(dim)
            for i in range(mu):
                mean += weights[i] * pop_sorted[i]
            mean_diff = mean - x0

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt.T @ (mean_diff / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff

            # Hessian update
            delta_hsig = (np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma) * np.linalg.norm(p_sigma)
            hsig = (delta_hsig < (1 + 6 / (dim + 1)) * np.sqrt((1 - (1 - c_sigma) ** (2 * (calls_made // pop_size)))) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // pop_size + 1))))
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_diff

            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            rank_one_factor = 1 + 2 * c_1 * (hsig - 1) if hsig else 1 - c_1

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = C_sqrt.T @ (pop_sorted[i] - x0) / sigma
                rank_mu += weights[i] * np.outer(z, z)
            rank_mu *= c_mu / sigma ** 2

            C = (1 + c_1 * rank_one_factor + c_mu - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Recompute C_sqrt via Cholesky (with fallback)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Update mean
            x0 = mean

        return best_x, best_f
# EVOLVE-BLOCK-END
