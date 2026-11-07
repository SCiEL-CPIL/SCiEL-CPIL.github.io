"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time limit guard: 300s limit, aim to finish in 280s
        start_time = time.time()
        time_limit = 280.0
        
        # CMA-ES Parameters
        # Initial population size: 4 + floor(3 * log(dim + 1))
        lam = 4 + int(3 * np.log(dim + 1))
        if lam < 4:
            lam = 4
        mu = lam // 2
        
        # Initial sigma
        sigma = 1.0 
        
        # Initial mean
        x_mean = np.zeros(dim)
        
        # Covariance matrix (Identity)
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu + 1.5) / (dim + 3.5 + mu)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2 / 3) - dim - 2) / ((dim + 2) ** 2 + mu))
        
        # Damping for covariance update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        
        # Chi-squared expectation for step size control
        chi_mu = np.sqrt(mu) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track best solution
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls_made = 1
        
        # Precompute weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Loop until budget or time limit
        while calls_made < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # Sample offspring
            offspring = []
            fitness = []
            
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                
            # Generate lam offspring
            for i in range(lam):
                if time.time() - start_time > time_limit:
                    break
                if calls_made >= budget:
                    break
                    
                z = np.random.randn(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)
                
            # If we couldn't generate any offspring due to time/budget, break
            if len(offspring) == 0:
                break
                
            # Evaluate offspring
            for y in offspring:
                fy = func(y)
                fitness.append(fy)
                calls_made += 1
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    
            # If we ran out of budget or time mid-generation, we might have fewer than lam offspring
            # We can still proceed with updates if we have at least mu offspring
            if len(offspring) < mu:
                break
                
            # Sort offspring by fitness
            indices = np.argsort(fitness)
            selected_indices = indices[:mu]
            
            # Update mean
            y_weighted = np.zeros(dim)
            for i, idx in enumerate(selected_indices):
                y_weighted += w[i] * offspring[idx]
                
            x_mean = x_mean + c_1 * (y_weighted - x_mean)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (y_weighted - x_mean)
            
            # Update p_c
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximation for the exponent part, using generation count
            gen_count = calls_made // lam
            exponent = 2 * gen_count
            denom = np.sqrt(1 - (1 - c_sigma) ** exponent)
            if denom == 0:
                denom = 1.0
            h_sigma = (norm_p_sigma / denom) < (1.4 + 2 / (dim + 1)) * chi_mu
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (y_weighted - x_mean)
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Outer product of selected offspring
            for i, idx in enumerate(selected_indices):
                diff = offspring[idx] - x_mean
                C += c_mu * w[i] * np.outer(diff, diff)
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_mu - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Symmetrize C
            C = (C + C.T) / 2.0

        return x_best, f_best
# EVOLVE-BLOCK-END
