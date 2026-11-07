import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for black-box continuous functions.
    Uses SciPy's L-BFGS-B which approximates the Hessian to take large,
    informed steps, highly effective for smooth functions like Griewank-Rosenbrock.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        # Fixed seed for reproducibility of the search path given the fixed function instance
        # This ensures consistent behavior if the evaluator doesn't seed numpy globally
        rng = np.random.RandomState(42)
        
        # Strategy:
        # 1. Evaluate a set of diverse starting points.
        # 2. Run L-BFGS-B from each starting point, constrained to the box [-5, 5].
        # 3. L-BFGS-B is very efficient for smooth functions, often converging in < 50-100 evals.
        # 4. With 500 budget, we can afford ~10-20 restarts.
        #    Let's use 20 restarts to ensure we cover the rotated/shifted landscape.
        #    500 / 20 = 25 evals per local optimization. 
        #    L-BFGS-B might use more, but we can cap it.
        #    Actually, L-BFGS-B maxiter is usually set to a number of iterations, not function evals.
        #    We need to wrap the function to count calls and stop if budget is exceeded.
        
        n_starts = 20
        
        # Generate starting points
        # Mix of center, random uniform, and maybe some structured points
        starts = []
        # 1. Center
        starts.append(np.zeros(dim))
        # 2. Random uniform in [-5, 5]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))
        
        # Wrap the function to track calls and enforce budget
        # The evaluator's func clips input, but we should ensure our inputs are valid 
        # or rely on its clipping. The contract says "func(x) clips a finite length-5 vector".
        # So we can pass any vector, it will be clipped.
        
        def wrapped_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget or time.time() > deadline:
                # Return a large value to stop optimization gracefully?
                # Or raise an exception? 
                # Raising an exception might be caught by scipy and treated as a failure,
                # but we want to stop cleanly.
                # Returning a large value might cause the optimizer to wander.
                # Better to just let it run and check budget in the loop?
                # No, L-BFGS-B calls func internally.
                # We can return the last known best value? No, that's misleading.
                # Let's just return a large penalty if we exceed budget, 
                # but the main loop should break.
                # Actually, if we exceed budget, the result is invalid.
                # We must stop BEFORE exceeding 500 calls.
                # So we check `if calls >= budget` and return a dummy value?
                # If we return a dummy value, the optimizer might continue.
                # We need to stop the optimizer.
                # SciPy's minimize doesn't have a callback to stop mid-iteration easily 
                # unless we raise an exception or return a specific value that triggers a stop condition?
                # No, the standard way is to use maxiter.
                # But maxiter is iterations, not function calls.
                # One iteration of L-BFGS-B can make multiple function calls (line search).
                # To be safe, we should limit the total function calls.
                
                # If we hit the budget limit, we should stop.
                # Returning a large value will likely cause the line search to fail 
                # and the iteration to terminate or take a tiny step.
                # Let's return a large value to discourage further exploration.
                return 1e10
                
            f_val = func(x)
            calls += 1
            
            if f_val < best_fx:
                best_fx = f_val
                best_x = x.copy()
                
            return f_val

        for start_idx in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = starts[start_idx]
            
            # Define bounds
            bounds = [(-5, 5)] * dim
            
            try:
                # L-BFGS-B parameters
                # maxiter: limit iterations to prevent runaway
                # maxls: limit line search steps
                # Since we have a tight budget, we should be conservative.
                # Let's allow up to 50 iterations per start? 
                # 20 starts * 50 iters * (1-2 func calls/iter) = 1000-2000 calls.
                # This might exceed budget.
                # Let's reduce n_starts or maxiter.
                # If we use 10 starts, we have 50 calls each.
                # Let's stick to 20 starts but limit maxiter to 20?
                # 20 * 20 * 2 = 800. Still risky.
                
                # Better approach: dynamic budget allocation.
                # If we have 500 calls, and L-BFGS-B is efficient, it might converge in 20-30 calls.
                # Let's set maxiter=30.
                # If it doesn't converge, it stops.
                
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20}
                )
                
                # Update best from result if it's better
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If an exception occurs, catch it and continue to next start
                # The contract says an exception later keeps the calls made.
                pass

        return best_x, best_fx
