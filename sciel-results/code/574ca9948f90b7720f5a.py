import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # CMA-ES Parameters
        pop_size = 10
        
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # Main loop with restarts
        while calls_made < budget and time.time() < deadline:
            # Initialize CMA-ES state
            if np.isinf(best_f):
                x_mean = np.random.uniform(-5, 5, dim)
            else:
                x_mean = best_x.copy()
            
            sigma = 2.0
            C = np.eye(dim)
            
            # CMA-ES adaptation parameters
            c_sigma = (pop_size + 2) / (dim + 4)
            c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 / ((dim + 2) ** 2 + pop_size))
            
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            best_f_in_run = best_f
            stagnation_count = 0
            stagnation_limit = 10
            
            # Inner CMA-ES loop
            while calls_made + pop_size <= budget and time.time() < deadline:
                # Generate population
                try:
                    w, v = np.linalg.eigh(C)
                    w = np.clip(w, 1e-20, None)
                    sqrt_w = np.sqrt(w)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    w, v = np.linalg.eigh(C)
                    w = np.clip(w, 1e-20, None)
                    sqrt_w = np.sqrt(w)
                
                batch_xs = []
                batch_fs = []
                
                for i in range(pop_size):
                    z = np.random.randn(dim)
                    x_new = x_mean + sigma * (v.T @ (sqrt_w * z))
                    f_new = evaluate(x_new)
                    if f_new is None:
                        break
                    batch_xs.append(x_new)
                    batch_fs.append(f_new)
                    
                if len(batch_xs) < pop_size:
                    break
                
                sort_idx = np.argsort(batch_fs)
                sorted_xs = [batch_xs[i] for i in sort_idx]
                sorted_fs = [batch_fs[i] for i in sort_idx]
                
                # Update best
                if sorted_fs[0] < best_f_in_run:
                    best_f_in_run = sorted_fs[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count >= stagnation_limit:
                    break
                
                mu = max(1, int(pop_size / 2))
                
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1 / np.sum(weights ** 2)
                
                x_old = x_mean
                x_mean = np.sum([weights[i] * sorted_xs[i] for i in range(mu)], axis=0)
                
                inv_sqrt_C = v.T @ np.diag(1.0 / sqrt_w) @ v
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                threshold_sigma = (1 + 0.3 * min(1, 1/mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_sigma * (2 - c_sigma))
                h_sigma = 1.0 if norm_p_sigma < threshold_sigma else 0.0
                
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (x_mean - x_old) / sigma
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                norm_p_c = np.linalg.norm(p_c)
                threshold_c = (1 + 0.3 * min(1, 1/mu_eff)) * (np.sqrt(mu_eff) + 1) * np.sqrt(c_c * (2 - c_c))
                h_c = 1.0 if norm_p_c < threshold_c else 0.0
                
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
                
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum([weights[i] * np.outer(sorted_xs[i] - x_old, sorted_xs[i] - x_old) for i in range(mu)], axis=0)
                
                # Simplified C update
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + rank_mu) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                if sigma < 1e-10:
                    break

        return best_x, best_f
