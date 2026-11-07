"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Separable CMA-ES implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        rng = np.random.default_rng(0)
        
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))
        sigma = 1.0
        x_mean = np.zeros(n)
        step_sizes = np.ones(n)
        
        # Parameters
        cs = (pop_size + 2) / (n + pop_size + 4)
        ds = 1 + 2 * max(0, (np.sqrt((pop_size + 1) / (n + 1)) - 1) / (1 + cs))
        c1 = (2) / ((n + 1.3) ** 2 + pop_size)
        cmu = min(1 - c1, 2 * ((pop_size - 2 + 1. / pop_size) / (n + 2)) ** 2 - 1)
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        weights = np.log(pop_size / 2) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mu_eff = 1. / np.sum(weights[:pop_size // 2] ** 2)
        
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1 / (21 * n ** 2))
        
        calls_used = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        fx = func(x_mean)
        calls_used += 1
        best_f = fx
        best_x = x_mean.copy()
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            y = rng.standard_normal((pop_size, n))
            x_offspring = x_mean + sigma * step_sizes * y
            
            f_offspring = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break
                x_i = np.clip(x_offspring[i], -5, 5)
                f_i = func(x_i)
                calls_used += 1
                f_offspring.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(f_offspring) < pop_size:
                break
                
            f_offspring = np.array(f_offspring)
            order = np.argsort(f_offspring)
            x_sorted = x_offspring[order]
            f_sorted = f_offspring[order]
            
            x_mean_new = np.sum(weights[:, np.newaxis] * x_sorted, axis=0)
            
            # Update p_sigma
            # For separable CMA-ES, the transformation is just division by step_sizes
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (x_mean_new - x_mean) / (sigma * step_sizes)
            
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - pop_size) / pop_size)) < (1.4 + 2 / (n + 1)) * chi_n
            
            # Update p_c
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_mean_new - x_mean) / (sigma * step_sizes)
            
            # Update step sizes
            # c1 and cmu adaptation for diagonal
            # delta_h = (1 - h_sigma) * cs * (2 - cs)
            # rank_one = outer(p_c, p_c)
            # rank_mu = sum(weights * outer(x_sorted - x_mean, x_sorted - x_mean)) / (sigma * step_sizes)^2
            # C_diag = (1 + c1*delta_h - c1 - cmu) * C_diag + c1 * rank_one + cmu * rank_mu
            # step_sizes = sqrt(C_diag)
            
            # Simplified update for step sizes:
            # Use the variance of the selected population
            # A more stable way:
            # Update C_diag directly
            C_diag = step_sizes ** 2
            delta_h = (1 - h_sigma) * cs * (2 - cs)
            rank_one = np.outer(p_c, p_c)
            
            # rank_mu calculation
            diffs = (x_sorted - x_mean) / (sigma * step_sizes)
            rank_mu = np.sum(weights[:, np.newaxis] * diffs ** 2, axis=0)
            
            C_diag = (1 + c1 * delta_h - c1 - cmu) * C_diag + c1 * rank_one + cmu * rank_mu
            # Ensure positivity
            C_diag = np.maximum(C_diag, 1e-10)
            step_sizes = np.sqrt(C_diag)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            x_mean = x_mean_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
