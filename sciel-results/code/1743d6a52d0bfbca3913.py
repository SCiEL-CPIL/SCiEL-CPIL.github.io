import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290  # Safety margin
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # CMA-ES parameters
        n = self.dim
        lam = 4 + int(3 * np.log(n))  # 8 for n=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        def cma_es_run(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f, start_time
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial population
            B, D = np.linalg.cholesky(C)
            inv_sqrt_D = 1.0 / np.sqrt(D)
            
            # We will simulate generations
            while calls_made < max_calls:
                if time.time() - start_time > time_limit:
                    return
                
                # Generate population
                Z = np.random.randn(lam, n)
                Y = x_mean + sigma * (Z @ (B * inv_sqrt_D).T)
                Y = np.clip(Y, -5, 5)
                
                # Evaluate
                F = np.array([func(y) for y in Y])
                calls_made += lam
                
                # Update best
                idx_min = np.argmin(F)
                if F[idx_min] < best_f:
                    best_f = F[idx_min]
                    best_x = Y[idx_min].copy()
                
                # Sort
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]
                
                # Recombination
                x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
                
                # Update mean
                x_mean = x_new
                
                # Update step size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (inv_sqrt_D * (Y[:mu].T @ weights).T)).T
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 1):
                    sigma *= np.exp(c_sigma / ccs * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam))) - np.sqrt(n)))
                    
                # Update covariance
                pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
                delta_sig = (1 - c_c) * (1 - c_c) # This is not quite right, standard formula is complex. 
                # For simplicity and robustness, let's use a simplified update or standard one.
                # Standard:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + delta_sig * C) + c_mu * sum(w_i (y_i - y_w)(y_i - y_w).T)
                
                # This implementation is getting long. 
                # Alternative: Use scipy's differential_evolution or dual_annealing? 
                # No, CMA-ES is better. 
                # Let's use a simpler, proven CMA-ES implementation structure.
                
                # If this gets too complex, fallback to Nelder-Mead with restarts is safer but worse.
                # However, the prompt says CMA-ES scored 4.459. I should try to implement it correctly.
                
                # Let's simplify: Use `scipy.optimize.minimize` with method='Nelder-Mead' or 'Powell' with multiple restarts?
                # Powell is good for box constraints and smooth-ish functions.
                # But f7 is step-like. Nelder-Mead might struggle with steps.
                
                # Let's try a different approach: **Population-based CMA-ES simplified** or **Covariance Matrix Adaptation via `scipy`?**
                # Actually, `scipy.optimize` does not have CMA-ES.
                
                # I will implement a standard CMA-ES carefully.
                
                # ... (Implementation continues in the final code block) ...
                break # Placeholder for logic flow in thought process

        # Main Loop
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, n)
        # Run CMA-ES
        # I will implement the full CMA-ES in the final code.
        
        # For the sake of the thought process, I assume the CMA-ES implementation works.
        # I need to handle the budget and time limits strictly.
        
        return best_x, best_f
