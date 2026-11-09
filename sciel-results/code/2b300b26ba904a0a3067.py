"""Rewritten optimizer using a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) approach."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # Number of recombination weights

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial mean and step size
        x_mean = np.zeros(dim)
        sigma = 1.0

        # Covariance matrix
        C = np.eye(dim)
        # Precompute Cholesky decomposition for sampling
        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Generate population
            pop = np.zeros((lam, dim))
            f_pop = np.zeros(lam)
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (C_inv_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                f_pop[i] = func(x_i)
                calls += 1
                
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x_i.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break

            # Sort population by fitness
            indices = np.argsort(f_pop)
            pop_sorted = pop[indices]
            f_sorted = f_pop[indices]

            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Update step size control
            x_diff = x_new_mean - x_mean
            sigma_new = sigma * np.exp(cs * (np.linalg.norm(x_diff) / (sigma * np.sqrt(2 * mueff)) - 1))
            
            # Update path evolution strategy vectors
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma_new * x_diff
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (calls // lam + 1)) < (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (calls // lam + 1))
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) / sigma_new * x_diff
            
            # Update covariance matrix
            C = (1 - (c1 + cmu)) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_sorted[:mu] - x_mean, pop_sorted[:mu] - x_mean), axis=0)
            
            # Ensure positive definiteness and update Cholesky
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset or regularize
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                sigma = 1.0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                break

            x_mean = x_new_mean
            sigma = sigma_new
            
            # Check for convergence
            if sigma < 1e-10 or np.allclose(C, np.eye(dim) * np.trace(C) / dim, atol=1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
