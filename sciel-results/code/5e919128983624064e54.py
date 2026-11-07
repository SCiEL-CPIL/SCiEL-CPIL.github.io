"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2  # Number of parents
        
        # Initial parameters
        x = np.zeros(dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        # CMA-ES state
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        cs = (mu + 2) / (dim + mu + 5)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu)
        cmu = min(1 - c1, 2 * (sqrt(mu_eff) - 2) / ((dim + 2) ** 2 + mu))
        
        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            fitnesses = []
            
            for i in range(lam):
                # Sample from N(x, sigma^2 * C)
                z = np.random.multivariate_normal(np.zeros(dim), C, size=1)
                y = x + sigma * z
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                population.append(y)
                fitnesses.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Select parents
            parents = population[:mu]
            
            # Update center of mass
            x_new = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            pc = (1 - cc) * pc + cc * (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C @ ((x_new - x) / sigma)
            
            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - lam + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            c_corr = 1 - (1 - hsig) * c1 * cs * (1 - cs)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * cs * (1 - cs) * C)
            for i in range(mu):
                diff = parents[i] - x
                C += cmu * weights[i] * np.outer(diff, diff)
            
            # Update step size
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = x_new
            
            # Re-evaluate center of mass to maintain best_x/best_f consistency if needed, 
            # but we already tracked best during generation.
            # Note: The CMA-ES mean x is updated, but we don't necessarily evaluate it 
            # unless it's part of the population or we want to be strict. 
            # Usually, the best individual from the population is tracked.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
