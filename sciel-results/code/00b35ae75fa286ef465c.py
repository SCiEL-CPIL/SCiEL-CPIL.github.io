import numpy as np
import scipy.optimize
import time

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
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        
        # Split budget: 40% for Global Search (ES), 55% for Local Refinement (NM), 5% for final polish
        es_budget = int(budget * 0.4)
        nm_budget = int(budget * 0.55)
        polish_budget = budget - es_budget - nm_budget
        
        # Ensure we have some budget for both phases
        if es_budget < 10:
            es_budget = max(1, budget // 3)
            nm_budget = budget - es_budget - polish_budget
            
        if nm_budget < 10 and budget > 30:
            nm_budget = 10
            es_budget = budget - nm_budget - polish_budget

        rng = np.random.default_rng(42)
        
        # --- PHASE 1: (1+1)-ES Global Search with improved adaptation ---
        # Start from a random point in the center
        x = rng.uniform(-3.0, 3.0, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.5
        calls_used = 1
        
        # Improved adaptive step size with momentum
        sigma_min = 0.005
        sigma_max = 4.0
        success_streak = 0
        failure_streak = 0
        
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
            
            # Update
            if fy < fx:
                x, fx = y, fy
                success_streak += 1
                failure_streak = 0
                # Increase sigma more aggressively on consecutive successes
                if success_streak >= 3:
                    sigma *= 1.15
                else:
                    sigma *= 1.05
            else:
                success_streak = 0
                failure_streak += 1
                # Decrease sigma on failure, more aggressively after consecutive failures
                if failure_streak >= 3:
                    sigma *= 0.85
                else:
                    sigma *= 0.95
            
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
        # Choose step size based on current sigma and problem scale
        simplex_step = max(sigma * 0.3, 0.05)
        simplex_step = min(simplex_step, 1.0)  # Cap to avoid too-large simplex
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
                    'xatol': 1e-12, 
                    'fatol': 1e-12, 
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except Exception:
            pass

        # --- PHASE 3: Final high-precision polish with L-BFGS-B ---
        if calls_used < budget and time.time() < deadline:
            polish_x = nm_best_x.copy() if nm_best_x is not None else best_x.copy()
            polish_f = nm_best_f if nm_best_f is not None else best_fx
            
            polish_calls = 0
            polish_best_x = polish_x.copy()
            polish_best_f = polish_f
            
            def polish_func(x):
                nonlocal polish_calls, polish_best_x, polish_best_f, calls_used
                
                if calls_used >= budget or time.time() > deadline:
                    return 1e6
                
                x = np.clip(x, lower, upper)
                
                try:
                    f_val = func(x)
                except Exception:
                    f_val = 1e6
                
                calls_used += 1
                polish_calls += 1
                
                if np.isnan(f_val) or np.isinf(f_val):
                    return 1e6
                
                if f_val < polish_best_f:
                    polish_best_f = f_val
                    polish_best_x = x.copy()
                    
                return f_val

            def polish_callback(xk):
                if calls_used >= budget or time.time() > deadline:
                    return True
                return False

            try:
                result2 = scipy.optimize.minimize(
                    polish_func,
                    polish_x,
                    method='L-BFGS-B',
                    bounds=[(lower, upper)] * dim,
                    options={
                        'maxiter': polish_budget * 2,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                    },
                    callback=polish_callback
                )
                if polish_best_f < nm_best_f:
                    nm_best_x = polish_best_x
                    nm_best_f = polish_best_f
            except Exception:
                pass

        # Final result is the best found in any phase
        if nm_best_x is not None and nm_best_f < best_fx:
            return nm_best_x, nm_best_f
        else:
            return best_x, best_fx
