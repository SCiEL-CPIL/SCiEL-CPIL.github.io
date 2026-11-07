import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0  # Safety margin

        # State
        evals = 0
        best_x = None
        best_f = np.inf

        # Helper to track evals and update best
        def tracked_func(x):
            nonlocal evals, best_x, best_f
            if time.time() - t_start > max_time:
                raise TimeoutError("Time limit exceeded")
            
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            
            f_val = func(x_clipped)
            evals += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Global Search (Random Sampling)
        # Allocate 20% of budget to global search
        global_budget = max(10, budget // 5)
        
        rng = np.random.default_rng()
        
        # Generate random points
        # We can generate them in batches
        for _ in range(global_budget):
            if evals >= budget:
                break
            x = rng.uniform(-5.0, 5.0, n)
            tracked_func(x)

        # Phase 2: Local Refinement (L-BFGS-B)
        # Use the remaining budget
        if evals < budget and best_x is not None:
            # L-BFGS-B requires bounds
            bounds = [(-5.0, 5.0)] * n
            
            # We need to limit the number of function calls in scipy
            # SciPy's minimize doesn't have a direct 'max_evals' for all methods,
            # but we can use a callback to stop it.
            
            stop_flag = [False]
            
            def callback(xk):
                nonlocal evals
                if evals >= budget:
                    stop_flag[0] = True
                    return True
                if time.time() - t_start > max_time:
                    stop_flag[0] = True
                    return True
                return False

            try:
                # maxiter is a safety net, callback is the primary control
                res = minimize(
                    tracked_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'callback': callback, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                # Update best if the result is better (it should be, but just in case)
                if res.success or (res.fun < best_f):
                    x_res = np.clip(res.x, -5.0, 5.0)
                    # The tracked_func already updated best_f during the run
                    # But res.x is the final point. We might have evaluated it already.
                    # We don't need to re-evaluate if tracked_func did it.
                    # However, tracked_func updates best_x/f inside.
                    # So best_x/best_f are already up to date.
                    
            except Exception as e:
                # If L-BFGS-B fails, we still have the best from Phase 1
                pass

        return best_x, best_f
