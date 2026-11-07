"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initial population
        # Start near center, as minimizer is in [-4, 4]^5
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        sigma = 2.0
        mean = x0.copy()
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 1)) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1) / ((self.dim + 2)**2 + mu_eff))
        
        best_x = mean.copy()
        best_f = f0
        
        calls_made = 1
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            B = np.linalg.cholesky(C)
            population = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                z = rng.standard_normal(self.dim)
                population[i] = mean + sigma * (B @ z)
                # Clip to box
                population[i] = np.clip(population[i], -5, 5)
            
            # Evaluate population
            fitness = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill remaining with best known to avoid issues, though we break
                    fitness[i] = best_f 
                    continue
                fitness[i] = func(population[i])
                calls_made += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = population[i].copy()
            
            # Sort by fitness
            idx = np.argsort(fitness)
            x_sorted = population[idx]
            f_sorted = fitness[idx]
            
            # Update mean
            mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update step size control variables
            y_mean = (mean - x0) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            h_sigma = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1) / self.budget)) <
                          (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1))) else 0
            
            # Update covariance matrix
            y_c = (mean - x0) / sigma
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            delta_heav = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(x_sorted[:mu] - x0, x_sorted[:mu] - x0), axis=0)
            
            C = (1 + c_1 * delta_heav - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(self.dim) - 1))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularization if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = np.diag(np.abs(np.diag(C)))
            
            # Update x0 to new mean for next iteration's relative calculation
            # Note: Standard CMA-ES uses mean as center. 
            # The 'x0' in the formulas above is the previous mean.
            x0 = mean
            
            # If sigma is too small, restart or stop
            if sigma < 1e-6:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
