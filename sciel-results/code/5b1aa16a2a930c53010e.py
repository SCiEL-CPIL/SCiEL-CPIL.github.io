"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-restart (1+1)-ES with L-BFGS-B polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Helper to check time and budget
        def should_stop():
            return calls >= budget or time.time() > deadline

        # Phase 1: Warm Start with random sampling
        # Use a larger initial population to find good starting basins
        n_warm = 50
        for _ in range(n_warm):
            if should_stop():
                break
            x = np.random.uniform(-5, 5, size=dim)
            f = func(x)
            calls += 1
            update_best(x, f)

        # Phase 2: Multi-restart (1+1)-ES with L-BFGS-B polish
        # We use a strategy of restarting the (1+1)-ES from the best point found
        # periodically, and using L-BFGS-B to polish the local minimum.
        
        # Parameters for (1+1)-ES
        # Step size adaptation
        c_sigma = 1.0 / (dim + 3)
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3)**2 + 1)
        
        # We will run several "cycles" of ES + Polish
        # Each cycle: run ES for a certain number of steps, then polish with L-BFGS-B
        # If polish improves, restart ES from the polished point with a smaller sigma
        
        num_cycles = 8
        es_steps_per_cycle = 30
        
        for cycle in range(num_cycles):
            if should_stop():
                break
            
            # Initialize ES from best point
            x0 = best_x.copy() if best_x is not None else np.zeros(dim)
            sigma = 1.5 / (1 + cycle * 0.5)  # Decrease initial sigma each cycle
            
            # State for (1+1)-ES
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            
            # Run ES for es_steps_per_cycle steps
            for step in range(es_steps_per_cycle):
                if should_stop():
                    break
                
                # Generate candidate
                z = np.random.standard_normal(dim)
                # Simple (1+1)-ES update: x_new = x0 + sigma * z
                # For better performance, we can use a covariance matrix, but let's keep it simple first
                # Actually, let's use a basic CMA-ES-like update for better convergence
                # For (1+1), we just sample from N(x0, sigma^2 I)
                y = x0 + sigma * z
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                # Update best
                update_best(y, fy)
                
                # Adapt step size
                if fy < best_f:
                    # Success: increase sigma slightly
                    sigma *= 1.1
                    # Move x0 towards y
                    x0 = y
                    # Update path memories (simplified)
                    pc = (1 - c_c) * pc + c_c * (y - x0 + sigma * z) / sigma
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma)) * z
                else:
                    # Failure: decrease sigma
                    sigma *= 0.8
                    # Keep x0 unchanged
                    pc = (1 - c_c) * pc
                    ps = (1 - c_sigma) * ps
                
                # Ensure sigma doesn't get too small or too large
                sigma = np.clip(sigma, 1e-4, 10.0)
            
            # Polish with L-BFGS-B
            if not should_stop() and best_x is not None:
                # Use L-BFGS-B to refine the best point
                # We need to track calls for L-BFGS-B
                def func_wrapper(x):
                    nonlocal calls
                    if calls >= budget or time.time() > deadline:
                        return best_f  # Return current best to stop
                    fx = func(x)
                    calls += 1
                    return fx
                
                # Set a callback to check time/budget
                def callback(xk):
                    return calls >= budget or time.time() > deadline
                
                try:
                    result = minimize(
                        func_wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8},
                        callback=callback
                    )
                    if result.success or result.fun < best_f:
                        update_best(result.x, result.fun)
                except Exception:
                    pass
        
        return best_x, best_f
