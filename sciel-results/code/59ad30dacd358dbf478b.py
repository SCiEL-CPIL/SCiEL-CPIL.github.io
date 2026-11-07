"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-restart Nelder-Mead optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Helper to evaluate and track best
        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
            return fx

        # Strategy: Multi-restart Nelder-Mead
        # We will run several instances of Nelder-Mead with different starting points.
        # To respect the strict budget, we wrap the function to count calls and stop early.
        
        # Initial set of starting points
        # 1. Origin (0,0,0,0,0) - often good for BBOB if shifts are small
        # 2. Random points within [-3, 3]
        # 3. Points near boundaries
        
        start_points = []
        
        # Origin
        start_points.append(np.zeros(dim))
        
        # Random points
        # Use a fixed seed for reproducibility within the run? No, task says "hidden seeded shift", 
        # but our optimizer can use its own randomness.
        # Let's use np.random with a seed derived from time or just default.
        # For stability in evaluation, maybe default random is fine.
        # We want diversity.
        
        # Generate 5 random starts
        for _ in range(5):
            x_start = np.random.uniform(-3.0, 3.0, dim)
            start_points.append(x_start)
            
        # Add some boundary-near points
        for _ in range(2):
            x_start = np.random.uniform(-4.5, -4.0, dim)
            start_points.append(x_start)
            
        for _ in range(2):
            x_start = np.random.uniform(4.0, 4.5, dim)
            start_points.append(x_start)

        # Shuffle starts to ensure variety if we stop early
        np.random.shuffle(start_points)

        # Total budget is 500.
        # We allocate budget per restart.
        # If we use 10 restarts, that's 50 calls each? That's very small for NM.
        # NM usually needs ~100-200 calls to converge in 5D.
        # Let's use fewer restarts but larger budgets.
        # 3 restarts with ~150 calls each? 
        # Or 5 restarts with 100 calls each?
        
        # Let's try 4 restarts.
        # Restart 1: 150 calls
        # Restart 2: 150 calls
        # Restart 3: 100 calls
        # Restart 4: 100 calls
        # Total: 500.
        
        # However, NM can converge early. We should stop a restart if it converges.
        
        num_restarts = 4
        # Define budget slices
        # We will pass a limit to each optimizer instance.
        # scipy.optimize.minimize doesn't have a strict "max function calls" parameter that stops exactly,
        # but we can wrap the function to raise an exception or return a specific value to stop?
        # Actually, `options={'maxiter': ...}` controls iterations, not function calls directly.
        # For Nelder-Mead, 1 iteration = 1 function call (roughly, actually it's 1 new point per iteration).
        # So maxiter ~ max function calls.
        
        # Let's define the budgets for each restart
        restart_budgets = [150, 150, 100, 100]
        
        # If we have more starts than restarts, we pick the first N.
        # If we have fewer, we repeat or add random.
        
        for i in range(num_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
                
            # Pick start point
            if i < len(start_points):
                x0 = start_points[i]
            else:
                x0 = np.random.uniform(-4.0, 4.0, dim)
                
            # Clip start point to valid box
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Determine remaining budget for this restart
            remaining_budget = budget - calls_used
            allocated = min(restart_budgets[i], remaining_budget)
            
            # We need a function wrapper that stops after `allocated` calls or if time is up.
            # We also want to record the best point found during this restart.
            
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = np.inf
            
            def wrapped_func(x):
                nonlocal local_calls, local_best_x, local_best_f
                if local_calls >= allocated:
                    # Raise an exception to stop the optimizer
                    # scipy handles exceptions poorly, better to just return a bad value?
                    # No, raising an exception is cleaner to stop.
                    # But we must catch it.
                    raise Exception("Budget Exceeded")
                
                local_calls += 1
                calls_used += 1
                
                # Check time
                if time.time() > deadline:
                    raise Exception("Time Exceeded")
                
                fx = func(x)
                if fx < local_best_f:
                    local_best_f = fx
                    local_best_x = x.copy()
                    
                # Also update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
                return fx

            try:
                # Use Nelder-Mead
                # Initial simplex size: 
                # Default in scipy is based on x0. 
                # We can provide initial simplex or use defaults.
                # For 5D, default is usually fine, but we can scale it.
                # Let's use standard options.
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': allocated,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # If it finished normally, update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception as e:
                # Stop if budget or time exceeded
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    break
                else:
                    # Other error, maybe break to be safe?
                    # Or continue to next restart?
                    # If scipy crashes for other reasons, it's better to stop and return what we have.
                    break

        # Final safety check
        if best_x is None:
            # Should not happen if budget > 0
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
