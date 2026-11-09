import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(n))
        if lambda_pop < 10: lambda_pop = 10
        if lambda_pop > 50: lambda_pop = 50
        
        mu = lambda_pop // 2
        
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        eval_count = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Pre-calculate Cholesky factor of C? No, C changes.
        # We need to sample from N(0, C).
        
        while eval_count < budget:
            if time.time() > deadline - 5:
                break
                
            # Generate population
            pop = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                # Sample z ~ N(0, I)
                z = np.random.standard_normal(n)
                # Transform to C
                # Use Eigen decomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                pop[i] = mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate
            fitness = np.array([func(x) for x in pop])
            eval_count += lambda_pop
            
            # Update best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_f:
                best_f = fitness[min_idx]
                best_x = pop[min_idx].copy()
                
            # Sort
            order = np.argsort(fitness)
            pop_sorted = pop[order]
            fitness_sorted = fitness[order]
            
            # Update Mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * pop_sorted[i]
                
            # Update Covariance Matrix Adaptation
            # Step direction
            step = (new_mean - mean) / sigma if sigma > 1e-10 else np.zeros(n)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
            
            # Update pc
            h_sigma = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) < (1 + 2 / n) * np.sqrt(np.trace(C)) / np.linalg.norm(np.sqrt(C)) else 0
            # Simplified h_sigma check
            ps_norm = np.linalg.norm(ps)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            h_sigma = 1 if ps_norm / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count // lambda_pop + 1))) < (1 + 2 / n) * chi_n else 0
            
            pc = (1 - c_c) * pc + h_sigma * c_c * step
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * (x_i - mean)(x_i - mean).T)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                diff = pop_sorted[i] - new_mean
                C_new += c_mu * weights[i] * np.outer(diff, diff)
            
            # Symmetrize
            C = (C_new + C_new.T) / 2
            
            # Update Sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            mean = new_mean
            
            # Re-check time
            if time.time() > deadline - 5:
                break
                
        return best_x, best_f
