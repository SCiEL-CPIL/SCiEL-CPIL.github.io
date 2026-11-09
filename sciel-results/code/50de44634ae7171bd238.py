import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(N))
        
        # Weights
        weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
        weights /= np.sum(weights)
        mu = int(lambda_ / 2)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        
        # Strategy parameters
        c_sigma = (2 + 2 / np.sqrt(N)) / (N + 4 + 2 / np.sqrt(N))
        c_c = (4 + lambda_ / mu_eff) / (N + 4 + 3 * lambda_ / mu_eff)
        c_1 = 2.0 / (N ** 2 + 18)
        c_mu = min(c_1 / 2, (mu_eff - 2 + (1.0 / (mu_eff - 1))) / (N ** 2 + 18))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (mu_eff + 1) - 1)) + c_sigma
        
        # Initial State
        x_mean = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        # Cholesky factor of C
        chol_C = np.linalg.cholesky(C)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Evaluate initial point
        f_val = func(x_mean)
        calls_made += 1
        if f_val < best_f:
            best_f = f_val
            best_x = x_mean.copy()
            
        # Main Loop
        while calls_made < budget:
            # Time Check
            if time.time() - start_time > time_limit:
                break
                
            # Generate population
            pop = []
            for i in range(lambda_):
                z = rng.standard_normal(N)
                x_i = x_mean + sigma * (chol_C @ z)
                # Clip to box [-5, 5]
                x_i = np.clip(x_i, -5, 5)
                pop.append(x_i)
                
            # Evaluate population
            f_values = []
            for x_i in pop:
                f_i = func(x_i)
                calls_made += 1
                f_values.append(f_i)
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
                    
            # Sort by fitness
            f_values = np.array(f_values)
            sorted_indices = np.argsort(f_values)
            pop = np.array(pop)
            pop_sorted = pop[sorted_indices]
            
            # Update Mean
            x_new_mean = np.sum(weights[:mu, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Update Step Size (Sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (chol_C.T @ (x_new_mean - x_mean) / sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(N) - 1))
            
            # Update Covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            C = (1 - c_1 * (2 - c_c)) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            # Simplified h_sigma check: if sigma changed significantly, we might skip rank-mu to avoid instability
            # But for small N, it's usually fine.
            diffs = (pop_sorted[:mu] - x_mean) / sigma
            C += c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->jk', diffs, diffs, diffs), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                chol_C = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(N)
                chol_C = np.linalg.cholesky(C)
            
            x_mean = x_new_mean
            
        return best_x, best_f
