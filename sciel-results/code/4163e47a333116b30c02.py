import numpy as np
import scipy.optimize
import time
from scipy.stats import norm

class Optimizer:
    """
    Hybrid Optimizer: Global (1+1)-ES with 1/5 success rule followed by Local Nelder-Mead refinement.
    
    Phase 1: (1+1)-ES with adaptive step size (1/5 success rule) to locate the global basin.
    Phase 2: Nelder-Mead to refine the solution to high precision.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        
        # Split budget: 50% for Global Search (ES), 50% for Local Refinement (NM)
        # The 1/5 success rule was observed to degrade performance, so we reduce ES time
        # and give more time to the high-precision NM phase which showed improvement.
        es_budget = int(budget * 0.5)
        nm_budget = budget - es_budget
        
        # Ensure we have some budget for both phases
        if es_budget < 10:
            es_budget = max(1, budget // 2)
            nm_budget = budget - es_budget
            
        if nm_budget < 10 and budget > 20:
            nm_budget = 10
            es_budget = budget - nm_budget

        rng = np.random.default_rng(42)
        
        # --- PHASE 1: (1+1)-ES Global Search (Simplified Adaptive) ---
        # Start from a random point in the center to avoid boundary bias initially
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        calls_used = 1
        
        # Simple adaptive step size: increase on success, decrease on failure
        # This avoids the complex 1/5 rule which might be too aggressive or slow to adapt
        sigma_min = 0.01
        sigma_max = 5.0
        success_count = 0
        total_attempts = 0
        
        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            # Generate offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, lower, upper)
            
            fy = func(y)
            calls_used += 1
            total_attempts += 1
            
            # Update
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                # Increase sigma on success
                sigma *= 1.1
            else:
                # Decrease sigma on failure
                sigma *= 0.9
            
            # Update global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            if sigma < sigma_min:
                sigma = sigma_min

        # --- PHASE 2: Nelder-Mead Local Refinement ---
        # Use the best point found in Phase 1 as the starting point for NM
        x_start = best_x.copy()
        
        # Create initial simplex for NM
        # Small initial step to ensure we are in the local basin
        simplex_step = max(sigma * 0.5, 0.1) 
        simplex = [x_start.copy()]
        for i in range(dim):
            p = x_start.copy()
            p[i] += simplex_step
            p = np.clip(p, lower, upper)
            simplex.append(p)
            
        # Wrapper for NM to count calls and enforce deadline
        nm_calls = 0
        nm_best_x = x_start.copy()
        nm_best_f = best_fx
        
        def wrapped_func(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            
            # Check budget and time
            if calls_used >= budget or time.time() > deadline:
                return 1e6
                
            x = np.clip(x, lower, upper)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = 1e6
                
            calls_used += 1
            nm_calls += 1
            
            if np.isnan(f_val) or np.isinf(f_val):
                return 1e6
                
            # Update local best for NM
            if f_val < nm_best_f:
                nm_best_f = f_val
                nm_best_x = x.copy()
                
            return f_val

        def callback(xk):
            # Stop if we hit budget or time
            if calls_used >= budget or time.time() > deadline:
                return True
            return False

        try:
            # Run Nelder-Mead with tighter tolerances for better precision
            result = scipy.optimize.minimize(
                wrapped_func, 
                x_start, 
                method='Nelder-Mead',
                options={
                    'maxiter': nm_budget * 2, # Allow enough iterations
                    'xatol': 1e-10, 
                    'fatol': 1e-10, 
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except Exception:
            pass

        # Final result is the best found in either phase
        if nm_best_x is not None:
            return nm_best_x, nm_best_f
        else:
            return best_x, best_fx
