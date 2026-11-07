"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin: 300s limit, leave 5s
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Wrapper to count calls
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            
            # Check if we have enough time for the rest of the optimization
            # Nelder-Mead can be slow if not careful, but with maxiter it's bounded.
            # However, we must stop if we exceed budget.
            
            fx = func(x_clipped)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
                
            return fx

        # Phase 1: Global Search / Multi-start
        # Use a set of starting points.
        # Since the minimizer is in [-4, 4]^5, we can sample from a Gaussian centered at 0 with std 2.0.
        # Or use a grid? Grid is expensive.
        # Let's use a mix of random points and a few structured points.
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in the algorithm's behavior, though evaluator might seed differently
        
        # We will run several Nelder-Mead optimizations from different starting points.
        # Each optimization will consume a portion of the budget.
        # Nelder-Mead with maxiter=100 uses roughly 100*dim evaluations? No, it's more like 100 iterations, each 1 eval, but it can be less.
        # Actually, scipy's Nelder-Mead maxiter is the max number of function evaluations? No, it's iterations.
        # In scipy.optimize.minimize, maxiter is the max number of iterations.
        # For Nelder-Mead, each iteration is roughly dim+1 evaluations? No, it's variable.
        # To be safe, we will limit the total evaluations per start.
        
        # Strategy:
        # 1. Try a few random starts.
        # 2. Try starts near the origin.
        # 3. Use the best result from previous starts as a new start (local refinement).
        
        # Number of starts: Let's aim for about 5-10 starts if budget allows.
        # With budget 500, if each NM run takes ~50 evals, we can do 10 runs.
        # But NM can take more. Let's limit maxiter for each run to 20.
        # 20 iterations * ~5 evals/iter = 100 evals? No, NM is usually 1 eval per iteration in scipy?
        # Checking scipy docs: "maxiter : int, optional. Maximum number of iterations to perform."
        # For Nelder-Mead, one iteration involves reflecting, expanding, contracting, shrinking.
        # It's safer to monitor calls_made inside the loop or use a tight maxiter.
        
        # Let's do a loop of starts.
        num_starts = 10
        max_iter_per_start = 30
        
        # Initial points
        # 1. Origin
        # 2. 5 random points in [-4, 4]
        # 3. 4 points at +/- 2 in each dimension (sparse grid corners)
        
        start_points = []
        start_points.append(np.zeros(dim))
        
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, dim))
            
        # Add some structured points
        for i in range(dim):
            p1 = np.zeros(dim)
            p2 = np.zeros(dim)
            p1[i] = 2.0
            p2[i] = -2.0
            start_points.append(p1)
            start_points.append(p2)
            
        # Shuffle to ensure diversity in early calls? 
        # Actually, deterministic order might be better for reproducibility if seed is fixed, 
        # but the evaluator seeds the *function*, not necessarily our rng. 
        # Wait, "hidden seeded shift...". The function is deterministic per instance.
        # Our rng seed 42 is fine.
        
        # Sort start points by distance to origin? Or just randomize?
        # Let's randomize the order of start points to avoid bias towards specific directions.
        idx = np.arange(len(start_points))
        rng.shuffle(idx)
        start_points = [start_points[i] for i in idx]
        
        # Run Nelder-Mead from each start point
        for i, x0 in enumerate(start_points):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10: # Leave some time for the final few calls
                break
                
            # Clip start point
            x0 = np.clip(x0, -5, 5)
            
            # We pass wrapped_func to minimize
            # Nelder-Mead parameters:
            # xatol, fatol: convergence tolerances
            # maxiter: max iterations
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter_per_start, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                # If it fails, continue to next start
                continue
            
            # After each run, we have updated best_x and best_f via wrapped_func
            # If we found a good point, we can restart from there with a smaller step size?
            # Nelder-Mead adapts step size. 
            # A "restart" from the best point with a fresh simplex might help escape local minima 
            # or refine further if the previous run stopped due to maxiter.
            
            # Check if we should do a refinement run from the current best
            if calls_made < budget - 20 and time.time() < deadline - 5:
                # Do one more run from the best point found so far
                # Use a tighter tolerance or just run again
                try:
                    res2 = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                    options={'maxiter': 20, 'xatol': 1e-5, 'fatol': 1e-5})
                except Exception:
                    pass

        # Final check: if we have budget left, do a random search to see if we missed a better region?
        # Unlikely for BBOB f7 which is unimodal (shifted/rotated ellipsoid).
        # The main risk is getting stuck in a poor local minimum if the function was multimodal, 
        # but f7 is Step Ellipsoidal, which is unimodal. 
        # So multi-start is mainly to ensure we find the basin of attraction.
        # Since it's unimodal, one good start might be enough, but multi-start is safer.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
