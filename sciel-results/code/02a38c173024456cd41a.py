"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and population sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        # For dim=5, lambda_pop = max(4, ceil(4 + 3*ln(5))) = max(4, ceil(4+4.83)) = max(4, 9) = 9

        # Initialize mean
        mean = np.zeros(dim)
        sigma = 3.0  # Initial step size

        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Success rate counter for step size adaptation
        successes = 0
        total_trials = 0
        target_success_rate = 1.0 / (lambda_pop * np.log(lambda_pop))

        # CMA-ES style covariance matrix (start with identity)
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = 1.0 / (lambda_pop + 3)
        c_c = (2 + 2.0 / (dim + 1)) / (lambda_pop + dim + 2)
        mu = lambda_pop // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while calls_made < budget:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                # Sample from multivariate normal with covariance C
                z = rng.multivariate_normal(np.zeros(dim), C, size=1).flatten()
                y = mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_offspring_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_offspring_f[0] < best_f:
                best_f = sorted_offspring_f[0]
                best_x = sorted_offspring[0].copy()

            # Compute new mean from top mu individuals
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_offspring[i]
            new_mean = np.clip(new_mean, -5, 5)

            # Update evolution paths
            diff = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            h_sig = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            C = (1 + c_c * (2 - c_c) * mu_eff - c_c) * C + c_c * np.outer(p_c, p_c)

            # Step size adaptation (1/5 rule variant)
            num_successes = 0
            for i in range(mu):
                if sorted_offspring_f[i] < best_f + 1e-10:
                    num_successes += 1
            # Use success rate for sigma adaptation
            success_rate = num_successes / mu
            if success_rate > target_success_rate:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 10.0)

            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
