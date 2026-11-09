import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        # 330s limit, use 325s as hard stop to be safe against overhead.
        time_limit = 325.0 

        # 1. Initialization: Scout for a good starting point
        # Evaluate 10 random points + center
        init_candidates = np.random.uniform(-5, 5, (9, n))
        init_candidates = np.vstack([np.zeros((1, n)), init_candidates])
        best_init_val = np.inf
        best_init_point = init_candidates[0]
        
        eval_count = 0
        for i in range(10):
            if time.time() - start_time > time_limit:
                break
            x = init_candidates[i]
            fx = func(x)
            eval_count += 1
            if fx < best_init_val:
                best_init_val = fx
                best_init_point = x.copy()
        
        x = best_init_point.copy()
        fx = best_init_val
        x_best = x.copy()
        f_best = fx
        
        # 2. (1+1)-ES with adaptive step size and restarts
        # Using gentler adaptation rates from Inspiration 2 (1.05/0.95) 
        # but keeping the robust restart structure from the current program.
        sigma = 1.5 # Initial step size
        sigma_min = 1e-10
        sigma_max = 8.0
        
        steps_since_best_improvement = 0
        restart_threshold = 40 # More patient before restart
        
        # Track multiple local optima for diversity (from Inspiration 2)
        # This helps in multimodal functions like Rastrigin/Gallagher
        local_optima = [(x_best.copy(), f_best)]
        max_local_optima = 5
        
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
            except Exception:
                break
                
            eval_count += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.05 # Gentler increase
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    steps_since_best_improvement = 0
                    # Update local optima: add the new best
                    local_optima.append((x_best.copy(), f_best))
                    # Sort by value and keep only the best 5
                    local_optima.sort(key=lambda t: t[1])
                    local_optima = local_optima[:max_local_optima]
                else:
                    steps_since_best_improvement += 1
            else:
                # Failure
                sigma *= 0.95 # Gentler decrease
                steps_since_best_improvement += 1
                
            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            if sigma < sigma_min:
                sigma = sigma_min
                
            # Restart mechanism
            # If we haven't improved the global best in 'restart_threshold' steps,
            # reset the local search to a random local optimum or the best point
            if steps_since_best_improvement >= restart_threshold:
                # Pick a random local optimum to escape local minima
                # With some probability, pick the best, otherwise pick a random one from the list
                if len(local_optima) > 1 and np.random.rand() < 0.5:
                    # Pick a random one from the stored optima
                    idx = np.random.randint(0, len(local_optima))
                    x = local_optima[idx][0].copy()
                    fx = local_optima[idx][1]
                else:
                    # Pick the best one
                    x = x_best.copy()
                    fx = f_best
                    
                sigma = 2.0 # Reset to a reasonable exploration size
                steps_since_best_improvement = 0

        # 3. Final Polish: Use remaining budget for a high-precision local search
        # This addresses the "final high-precision polish pass" directive.
        # We use a smaller sigma and more iterations to fine-tune the best point found.
        if eval_count < budget and time.time() - start_time < time_limit:
            # Use a fraction of remaining budget for polish
            remaining = budget - eval_count
            polish_budget = min(remaining, 50) # Cap polish at 50 steps
            polish_sigma = max(sigma, 0.1) * 0.1 # Start with smaller sigma
            x_polish = x_best.copy()
            f_polish = f_best
            
            for i in range(polish_budget):
                if time.time() - start_time > time_limit:
                    break
                if eval_count >= budget:
                    break
                    
                z = np.random.standard_normal(n)
                y = x_polish + polish_sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                    
                eval_count += 1
                
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    polish_sigma *= 1.05 # Slightly increase if successful
                else:
                    polish_sigma *= 0.9 # Decrease if failed
                    
                if polish_sigma < sigma_min:
                    polish_sigma = sigma_min
                    
            if f_polish < f_best:
                f_best = f_polish
                x_best = x_polish.copy()
                
        return x_best, f_best
