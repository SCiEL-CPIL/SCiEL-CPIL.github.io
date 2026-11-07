import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # Multi-start CMA-ES
        while calls < budget and time_ok():
            # Initialize a new CMA-ES instance
            # Random start point in the box
            mean = np.random.uniform(-4.5, 4.5, d)
            
            # Initial sigma: large enough to explore, small enough to be efficient
            sigma = 1.5
            
            # CMA-ES parameters
            lam = 12
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c1 = 2 / ((d + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
            
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            local_best_f = float('inf')
            local_best_x = mean.copy()
            no_improve_count = 0
            gen = 0
            
            # Run CMA-ES until stagnation or budget/time exhausted
            while calls < budget and time_ok():
                # Generate population
                pops = []
                for i in range(lam):
                    if calls >= budget or not time_ok():
                        break
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        C = C + 1e-10 * np.eye(d)
                        L = np.linalg.cholesky(C)
                    
                    z = np.random.randn(d)
                    y = mean + sigma * L @ z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    pops.append((y, fy))
                    
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if len(pops) < mu:
                    break
                
                # Sort
                pops.sort(key=lambda t: t[1])
                
                # Update mean
                old_mean = mean.copy()
                mean = sum(w[i] * pops[i][0] for i in range(mu))
                
                # Update sigma
                delta = mean - old_mean
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-16:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta / delta_norm
                    h_sigma = np.linalg.norm(p_sigma) / (np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) * np.sqrt(1 - c_sigma))
                    chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d*d))
                    sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma)/chi_d - 1))
                
                # Update C
                delta_c = (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_c
                
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) / np.sqrt(1 - c_c)) < (1 + 2/(d+1)) * np.sqrt(d) * (1 + 1/(d+1))
                
                rank1 = np.outer(delta_c, delta_c)
                rankmu = sum(w[i] * np.outer(pops[i][0] - old_mean, pops[i][0] - old_mean) / (sigma**2) for i in range(mu))
                
                C = (1 - c1 - cmu + (1 - h_c) * c_c * (2 - c_c)) * C + c1 * rank1 + cmu * rankmu
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    np.linalg.inv(C)
                except:
                    C = C + 1e-10 * np.eye(d)
                
                gen += 1
                
                if sigma < 1e-14:
                    break
                    
                # Stagnation check: restart if no improvement for 15 evaluations
                if no_improve_count >= 15:
                    # Restart with a new random point
                    # If we found a good point, we might want to bias the new start towards it?
                    # For Discus, the function is shifted. A global restart is safer.
                    # But let's try a local restart around the current best if it's very good?
                    # No, standard multi-start usually does global or local.
                    # Let's do a local restart with a larger sigma if we've converged to a local minimum.
                    # Actually, for BBOB f11 (Discus), the minimum is shifted.
                    # If we stagnate, we are likely stuck in a local minimum or the search is too small.
                    # Let's restart with a new random point to explore other areas.
                    break
        
        return best_x, best_f
