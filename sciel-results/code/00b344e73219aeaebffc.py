import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline
        
        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # Multi-start: restart if no improvement for this many evaluations
        stagnation_limit = 15
        
        while calls < budget and time_ok():
            # Initialize CMA-ES state
            mean = np.random.uniform(-4, 4, d)  # Start in [-4, 4]^5 where minimizer lies
            sigma = 1.5
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            gen = 0
            no_improve_count = 0
            local_best_f = float('inf')
            
            # Evaluate initial point
            if calls < budget and time_ok():
                f_init = func(mean)
                calls += 1
                if f_init < best_f:
                    best_f = f_init
                    best_x = mean.copy()
                local_best_f = f_init
            
            # Run CMA-ES until stagnation or budget/time
            while calls < budget and time_ok() and no_improve_count < stagnation_limit:
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
                        no_improve_count = 0
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if len(pops) < mu:
                    break
                
                # If no improvement this generation
                if no_improve_count == 0 and local_best_f == float('inf'):
                    no_improve_count = 1
                # Check if we had improvement this generation
                gen_had_improvement = False
                for _, fy in pops:
                    if fy < local_best_f:
                        gen_had_improvement = True
                        break
                if not gen_had_improvement:
                    no_improve_count += 1
                
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
                    chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d*d))
                    sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma)/chi_d - 1))
                
                # Update C
                delta_c = (mean - old_mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_c
                
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) / np.sqrt(1 - c_c)) < (1 + 2/(d+1)) * np.sqrt(d) * (1 + 1/(d+1))
                
                rank1 = np.outer(delta_c, delta_c)
                rankmu = sum(w[i] * np.outer(pops[i][0] - old_mean, pops[i][0] - old_mean) / (sigma**2) for i in range(mu))
                
                C = (1 - c1 - cmu + (1 - h_c) * c_c * (2 - c_c)) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                try:
                    np.linalg.inv(C)
                except:
                    C = C + 1e-10 * np.eye(d)
                
                gen += 1
                
                if sigma < 1e-14:
                    break
        
        return best_x, best_f
