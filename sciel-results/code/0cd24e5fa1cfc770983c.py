import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def cma_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Strategy parameters
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
            
            # State
            m = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(20*n**2))
            
            # Initial population
            # We will generate candidates one by one to count calls precisely
            
            # Sort indices for selection (pre-compute if possible, but weights change? No, weights are fixed)
            # Actually, we need to sort the fitness values each generation.
            
            for gen in range(max_calls // lam + 1):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                
                # Generate population
                # We need to evaluate lam points.
                # To save calls, we can stop early if we run out of budget mid-generation?
                # Better to just check budget before each evaluation.
                
                y_points = []
                f_points = []
                
                for i in range(lam):
                    if calls_used >= budget:
                        return
                    if time.time() > deadline:
                        return
                        
                    z = rng.standard_normal(n)
                    y = m + sigma * z
                    # Clip to box
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    calls_used += 1
                    
                    y_points.append(y)
                    f_points.append(fy)
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                # Sort by fitness
                order = np.argsort(f_points)
                y_sorted = np.array(y_points)[order]
                f_sorted = np.array(f_points)[order]
                
                # Update mean
                m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # Hessian of the quadratic form
                y_diffs = (y_sorted[:mu] - m) / sigma
                # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cs) * (1 - cs) * cs / (n + cs) * (1 + 2/mueff * (n - 1) if ... )) * ps * ps.T) + cmu * sum(weights * y_diffs * y_diffs.T)
                # Simplified update:
                hsig = (np.sqrt(1 - (1 - cs)**2 * gen / 10)) * (np.linalg.norm(ps) / chi_n < 1.4 + 2/(n+1))
                
                pc = (1 - cs) * pc + hsig * cs * mueff * y_diffs.mean(axis=0)
                
                delta_h = (1 - hsig) * (1 - (1 - cs)**2 * (gen + 1) / 10)
                C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cs) * (1 - cs) * cs / (n + cs) * (1 + 2/mueff * (n - 1) if hsig else 0) ) * ps * ps.T) + cmu * np.sum(weights[:, None, None] * np.outer(y_diffs, y_diffs), axis=0)
                
                # Note: The above C update is a bit complex. A simpler rank-1 and rank-mu update is standard.
                # Let's use a simpler, robust update formula often found in tutorials.
                # Actually, let's just use a simple (1+1)-ES with restarts if CMA-ES is too complex to get right quickly.
                # But CMA-ES is much better.
                
                # Let's stick to a simpler (1+1)-ES with restarts, as it's less prone to implementation bugs in a short script.
                # The parent score 1.0413 suggests a simple ES worked well.
                pass

        # Given the risk of CMA-ES implementation errors, let's go with a highly optimized Multi-Start (1+1)-ES.
        
        def run_1plus1_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            x = x0.copy()
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = sigma0
            no_improve = 0
            max_no_improve = 50
            
            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.1 # Increase step on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.9 # Decrease step on failure
                    if no_improve >= max_no_improve:
                        # Restart logic: try a larger jump from current best
                        # Or break to let the outer loop handle restarts
                        break

        # Strategy:
        # 1. Start at origin with moderate sigma.
        # 2. If stuck, restart from best with larger sigma.
        # 3. Start from random points if budget allows.
        
        # Phase 1: Origin start
        x0 = np.zeros(dim)
        run_1plus1_es(x0, 1.5, int(budget * 0.4))
        
        # Phase 2: Restart from best with larger sigma if needed
        if calls_used < budget and best_x is not None:
            # Check if we are stuck (sigma might be small)
            # Just run another ES from best_x with a fresh, larger sigma
            run_1plus1_es(best_x, 2.0, int(budget * 0.3))
            
        # Phase 3: Random starts
        remaining = budget - calls_used
        if remaining > 50:
            # Split remaining among 2 random starts
            calls_per_start = remaining // 2
            for _ in range(2):
                x0 = rng.uniform(-4, 4, dim)
                run_1plus1_es(x0, 1.0, calls_per_start)
                if calls_used >= budget:
                    break

        return best_x, best_f
