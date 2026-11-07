"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size and boundary handling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Population size: 4 + floor(3 * log(dim + 1))
        lam = max(4, int(4 + 3 * np.log(dim + 1)))
        mu = lam // 2  # number of parents

        # Step size
        sigma = 1.0

        # Covariance matrix
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        c_sigma = (mu + 0.5) / (dim + 2)
        d_sigma = 1 + 2 * max(0, (np.sqrt((mu + 1) / (dim + 1)) - 1) / (dim + 2)) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Track calls
        calls = 1

        while calls < budget:
            # Generate population
            # Update sigma using success rule (simplified)
            # For CMA-ES, we typically update after selection
            
            # Generate offspring
            offspring = []
            for i in range(lam):
                # Sample from N(x, sigma^2 * C)
                # Use Cholesky decomposition for efficiency
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                
                offspring.append((fy, y))
                
                if calls >= budget:
                    break

            if len(offspring) < mu:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Selection
            selected = [y for _, y in offspring[:mu]]
            
            # Weighted mean
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * selected[i]
            
            # Update evolution paths
            if sigma > 0:
                diff = (m - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1))) if calls > 0 else 1.0
                # Simplified h_sigma check
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) - 1 else 0.0
                
                p_c = (1 - c_c) * p_c + h_sigma * c_c * diff
                C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            # Update mean
            x = m
            fx = best_fx  # We don't have the function value of m directly, but we track best

            # Ensure x stays in bounds
            x = np.clip(x, -5, 5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
