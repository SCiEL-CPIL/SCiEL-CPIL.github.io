import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: aim to finish well before 330s.
        start_time = time.time()
        time_limit = 290.0  # Safety margin

        # CMA-ES Parameters
        # For dim=5 and budget=500, a population of ~10 is reasonable.
        # lambda = 4 + floor(3 * log(dim)) = 4 + 3*1.609 = 8. Let's use 10 for better exploration.
        lam = 10
        
        # Initial step size. Domain [-5, 5].
        sigma = 2.0
        
        # Initial mean.
        mean = np.zeros(dim)
        
        # Initial covariance
        C = np.eye(dim)
        
        # Cumulation paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (dim + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial evaluation
        x_best = mean.copy()
        f_best = func(x_best)
        calls_made = 1
        
        # Pre-allocate population storage
        population = np.zeros((lam, dim))
        fitness = np.zeros(lam)
        
        # Helper to compute C_sqrt and C_inv_sqrt
        def update_matrices(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            inv_sqrt_eigvals = 1.0 / sqrt_eigvals
            C_sqrt = (eigvecs * sqrt_eigvals) @ eigvecs.T
            C_inv_sqrt = (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
            return C_sqrt, C_inv_sqrt

        C_sqrt, C_inv_sqrt = update_matrices(C)
        
        while calls_made < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # Generate offspring
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = np.random.randn(dim)
                y = sigma * (C_sqrt @ z)
                x = mean + y
                x = np.clip(x, -5, 5)
                population[i] = x
                fitness[i] = func(x)
                calls_made += 1
                
                if fitness[i] < f_best:
                    f_best = fitness[i]
                    x_best = x.copy()
                    
            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            fitness_sorted = fitness[idx]
            population_sorted = population[idx]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * population_sorted[:mu], axis=0)
            
            # Update step size
            y_k = (population_sorted[:mu] - mean_old) / sigma
            y_mean = np.sum(weights[:, None] * y_k, axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_mean)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1 if (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            pc = (1 - c_c) * pc + h_sigma * c_c * (mean - mean_old) / sigma
            
            # Update Covariance C
            sum_yy = np.zeros((dim, dim))
            for i in range(mu):
                sum_yy += weights[i] * np.outer(y_k[i], y_k[i])
                
            pc_outer = np.outer(pc, pc)
            correction = (1 - h_sigma) * c_c * (2 - c_c) * pc_outer
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc_outer + correction) + c_mu * sum_yy
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update matrices
            C_sqrt, C_inv_sqrt = update_matrices(C)
            
        return x_best, f_best
