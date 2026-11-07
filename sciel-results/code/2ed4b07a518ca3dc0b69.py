"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-start L-BFGS-B optimizer with Sobol initialization and Powell polish for BBOB f11 (Discus)."""

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

        calls_made = 0
        best_x = None
        best_f = np.inf

        # Wrapper to track calls and enforce budget
        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                # Return a large value to discourage further exploration, 
                # but we should ideally stop the optimizer. 
                # However, minimize might not stop immediately.
                # We rely on maxfun/maxiter limits primarily.
                return 1e100
            calls_made += 1
            return func(x)

        # Phase 1: Initial Sampling using Sobol sequence for better coverage
        # 20 points is a good balance for d=5
        n_init = 20
        sobol = qmc.Sobol(d=d, scramble=True, seed=42)
        # Generate points in [0, 1)^d
        samples = sobol.random(n_init)
        # Map to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Add origin as a specific candidate, just in case
        samples = np.vstack([np.zeros(d), samples])
        
        # Evaluate initial samples
        initial_results = []
        for x in samples:
            if not time_ok() or calls_made >= budget:
                break
            fx = safe_func(x)
            initial_results.append((fx, x.copy()))
            
        if not initial_results:
            # Fallback if we couldn't even make the first call (should not happen)
            best_x = np.zeros(d)
            best_f = safe_func(best_x)
            return best_x, best_f

        # Sort initial results by function value
        initial_results.sort(key=lambda t: t[0])
        
        # Update best
        best_f = initial_results[0][0]
        best_x = initial_results[0][1]

        # Phase 2: Multi-start L-BFGS-B
        # We will try the top few distinct starting points
        # Define a minimum distance to consider a start "distinct"
        min_dist = 1.0 
        
        starts_to_try = []
        for fx, x in initial_results:
            if len(starts_to_try) >= 5: # Max 5 restarts
                break
            # Check if this point is distinct from already selected starts
            is_distinct = True
            for _, x_prev in starts_to_try:
                if np.linalg.norm(x - x_prev) < min_dist:
                    is_distinct = False
                    break
            if is_distinct:
                starts_to_try.append((fx, x))
        
        # If we didn't get 5 distinct points, just use the top ones we have
        if len(starts_to_try) < 3:
             for fx, x in initial_results:
                 if (fx, x) not in starts_to_try:
                     starts_to_try.append((fx, x))
                 if len(starts_to_try) >= 3:
                     break

        # Allocate budget for each start
        # We have used ~21 calls so far.
        # Remaining: budget - calls_made
        # Let's give the first (best) start the most budget, decreasing for subsequent ones.
        # L-BFGS-B is efficient, so it might converge in 20-50 calls.
        # But if it struggles, we want to move on.
        
        for i, (fx_init, x0) in enumerate(starts_to_try):
            if not time_ok() or calls_made >= budget:
                break
            
            # Calculate remaining budget
            remaining = budget - calls_made
            if remaining < 10:
                break
                
            # Determine maxfun for this start
            # First start gets up to 200 calls, second 100, third 50, etc.
            # But capped by remaining budget.
            if i == 0:
                max_fun = min(200, remaining - 10) # Leave 10 for polish
            elif i == 1:
                max_fun = min(100, remaining - 10)
            elif i == 2:
                max_fun = min(50, remaining - 10)
            else:
                max_fun = min(30, remaining - 10)
                
            if max_fun < 5:
                break

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={
                        'maxiter': 100,
                        'maxfun': max_fun,
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                
                # safe_func already updated calls_made
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
                # If we found a very good solution, we can stop early
                # The minimum is in [-100, 100]. If we are below -50, we are likely close.
                if best_f < -50:
                    # Don't break immediately, maybe one more polish is worth it, 
                    # but let's save budget for polish.
                    pass

            except Exception:
                pass

        # Phase 3: Final Polish
        # Use remaining budget for a high-precision polish on the best point found.
        if calls_made < budget and time_ok() and best_x is not None:
            remaining = budget - calls_made
            if remaining > 5:
                # Try Powell method for derivative-free polishing in the narrow valley
                # Powell is robust to the shape of the valley.
                try:
                    # Limit maxiter to avoid using all budget if it's slow
                    max_iter = min(remaining - 2, 100)
                    result = minimize(
                        safe_func,
                        best_x,
                        method='Powell',
                        bounds=[(-5, 5)] * d,
                        options={
                            'maxiter': max_iter,
                            'xtol': 1e-8,
                            'ftol': 1e-12
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
