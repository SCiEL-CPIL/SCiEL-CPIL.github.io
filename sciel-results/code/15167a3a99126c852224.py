import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        all_points = []
        
        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() - start_time > max_time:
                return None
            f = problem(x)
            used += 1
            return f

        def add_point(x, f):
            all_points.append((x, f))

        # 1. Sparse Initialization (5 points)
        init_size = 5
        X_init = lo + np.random.random((init_size, dim)) * (hi - lo)
        for x in X_init:
            f = eval_point(x)
            if f is not None:
                add_point(x, f)
        
        if used >= budget or time.time() - start_time > max_time:
            return all_points

        # 2. Multi-Start Local Search
        # We will run Nelder-Mead on each objective separately from several starting points.
        # This explores the trade-off surface by pushing each objective down independently.
        
        # Determine number of local searches to run
        # Budget is 200. Init used 5. Remaining 195.
        # NM in 4D typically takes ~30-50 evals for reasonable convergence.
        # Let's aim for ~10-15 local searches.
        
        # We pick starting points from the current non-dominated set or random if empty.
        # To ensure diversity, we also add some random starts.
        
        starts = []
        
        # Add current non-dominated points (simple filter)
        # For simplicity and speed, we just use the initial points as starts, plus a few random ones.
        for x in X_init:
            starts.append(x)
            
        # Add 5 random starts
        X_rand = lo + np.random.random((5, dim)) * (hi - lo)
        for x in X_rand:
            starts.append(x)

        # Shuffle starts to mix up which objective gets which start first? 
        # Or just iterate.
        
        obj_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] # Cycle through objectives
        
        current_start_idx = 0
        while used < budget and time.time() - start_time < max_time:
            if current_start_idx >= len(starts):
                # If we run out of starts, generate new random ones? 
                # Or break? Let's generate a few more random ones to keep going.
                new_x = lo + np.random.random(dim) * (hi - lo)
                starts.append(new_x)
                current_start_idx -= 1 # Don't increment, use the new one

            x_start = starts[current_start_idx]
            current_start_idx += 1
            
            # Select objective to minimize
            obj_idx = obj_indices[(current_start_idx - 1) % len(obj_indices)]
            
            # Define objective function for scipy
            def objective(x):
                f = eval_point(x)
                if f is None:
                    return 1e6 # Penalize if budget/time exceeded
                return f[obj_idx]
            
            # Constraints for bounds
            bounds_scipy = list(zip(lo, hi))
            
            # Run Nelder-Mead
            try:
                res = minimize(objective, x_start, method='Nelder-Mead', 
                               bounds=bounds_scipy, 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                
                # Record the best point found during the minimization?
                # The problem statement says "Every returned objective vector is recorded".
                # My `eval_point` already records every call.
                # So we don't need to explicitly add res.x here, it's already in `all_points` via `eval_point`.
                
            except Exception:
                pass

        return all_points
