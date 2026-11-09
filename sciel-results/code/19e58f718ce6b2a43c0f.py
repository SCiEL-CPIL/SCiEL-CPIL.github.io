import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
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
        
        def cma_es(x_start, sigma_init, max_evals, max_generations):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_x, best_f
            
            pop_size = 5
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            
            c_sigma = (pop_size + 2) / (dim + 4)
            c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 / ((dim + 2) ** 2 + pop_size))
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_f = float('inf')
            local_best_x = x_mean.copy()
            stagnation_count = 0
            stagnation_limit = 8
            
            gen = 0
            while calls_made + pop_size <= budget and gen < max_generations and time.time() < deadline:
                gen += 1
                
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
                
                if sorted_fs[0] < local_best_f:
                    local_best_f = sorted_fs[0]
                    local_best_x = sorted_xs[0].copy()
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
                
                C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + rank_mu) + c_mu * rank_mu
                C = (C + C.T) / 2
                
                if sigma < 1e-10:
                    break
            
            return local_best_x, local_best_f
        
        def lbfgsb_polish(x_start, max_evals):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return x_start, best_f
            
            remaining = budget - calls_made
            if remaining < 10:
                return x_start, best_f
            
            bounds = [(-5, 5)] * dim
            
            def objective(x):
                return evaluate(x)
            
            try:
                result = minimize(objective, x_start, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': min(remaining, 50), 'maxls': 20})
                if result.success and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            return best_x, best_f
        
        # Phase 1: Global CMA-ES exploration
        x_start = np.random.uniform(-4, 4, dim)
        x_cma, f_cma = cma_es(x_start, 3.0, 300, 40)
        
        # Phase 2: L-BFGS-B polish
        x_polished, f_polished = lbfgsb_polish(x_cma, 50)
        
        # Phase 3: Second CMA-ES from best if budget remains
        if calls_made < budget - 50 and time.time() < deadline:
            x_start2 = best_x + np.random.randn(dim) * 0.5
            x_cma2, f_cma2 = cma_es(x_start2, 1.0, budget - calls_made - 50, 20)
            x_polished2, f_polished2 = lbfgsb_polish(x_cma2, 50)
        
        # Final polish
        if calls_made < budget - 10 and time.time() < deadline:
            lbfgsb_polish(best_x, 20)
        
        return best_x, best_f
