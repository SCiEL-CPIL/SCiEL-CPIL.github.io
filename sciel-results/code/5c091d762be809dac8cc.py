import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin

        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def wrapper(x):
            nonlocal calls_made
            calls_made += 1
            if calls_made > budget:
                # This shouldn't happen if we manage maxfev correctly, but just in case
                return 1e10
            return func(x)

        bounds = [(-5, 5)] * dim
        
        # Phase 1: Global Search (Random Sampling)
        # Use 10% of budget or at least 10 calls for global exploration
        n_global = max(10, budget // 10)
        
        # We will do random sampling manually to control the call count precisely
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            f = wrapper(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapper(best_x)

        # Phase 2: Local Search with L-BFGS-B
        # We will use the remaining budget for L-BFGS-B runs.
        # Strategy: Run L-BFGS-B from the current best. If it converges or stalls, 
        # try a new random start or a perturbed start.
        
        remaining_budget = budget - calls_made
        
        # To ensure we don't get stuck in a local minimum (though Rosenbrock is unimodal in the main basin),
        # we can try multiple starts.
        # Given the budget is 500, and L-BFGS-B is efficient, we can afford several runs.
        
        # Let's define a helper to run L-BFGS-B with a specific maxfev
        def run_lbfgsb(start_x, maxfev):
            nonlocal best_x, best_f, calls_made
            if maxfev <= 0:
                return
            
            # L-BFGS-B is a gradient-based method. It requires gradients.
            # scipy.optimize.minimize with L-BFGS-B uses numerical gradients if not provided.
            # For 5D, numerical gradients are fine (5 extra calls per eval? No, finite difference is 2N or N+1 calls).
            # Actually, L-BFGS-B in scipy uses forward differences for gradient estimation if jac is not provided.
            # This might be expensive. 
            # Let's check if we can provide a gradient. The task says func is a black box.
            # So we must rely on numerical gradients.
            # For dim=5, one gradient evaluation costs ~6 function calls (forward diff).
            # This is acceptable.
            
            res = minimize(
                wrapper, 
                start_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'maxfev': maxfev}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
            return res

        # Loop until budget is exhausted or time runs out
        # We'll do a series of L-BFGS-B runs.
        # First run from the best random point.
        # Then, if we have budget left, try other random points or refine the best.
        
        # Since Rosenbrock has a single global minimum (in the standard definition, rotated here),
        # finding the basin is key. The random phase should do it.
        # L-BFGS-B should converge very quickly once in the basin.
        
        # We will allocate the remaining budget in chunks.
        # Chunk size: 50 calls.
        chunk_size = 50
        
        # List of starting points to try
        # 1. Current best
        # 2. Other good points from random phase? We didn't store them.
        # Let's just generate new random points if the current best doesn't improve much.
        
        # Actually, a simpler strategy:
        # Run L-BFGS-B from best_x with remaining budget.
        # If it converges (success=True) or reaches maxfev, check if we have budget left.
        # If we have budget left and best_f is not "small enough" (e.g. > 1e-4), try a new random start.
        
        while calls_made < budget and time.time() < deadline:
            # Determine maxfev for this run
            # Leave 1 call in reserve?
            maxfev = min(remaining_budget, chunk_size)
            
            if maxfev <= 0:
                break
            
            # Start from best_x
            # To avoid getting stuck in a local optimum (if any) or if the basin was missed,
            # we could mix in random starts.
            # But for Rosenbrock, if we are in the basin, we are in.
            # If we are not in the basin, L-BFGS-B might go to a different local min or fail.
            
            # Let's try to be smart:
            # If best_f is high (e.g. > 10), we might be in a bad spot. Try a new random start.
            # If best_f is low, refine.
            
            start_x = best_x
            if best_f > 10.0 and np.random.rand() < 0.5:
                # 50% chance to try a new random start if we are stuck high
                start_x = np.random.uniform(-5, 5, dim)
            
            run_lbfgsb(start_x, maxfev)
            
            # Update remaining budget
            remaining_budget = budget - calls_made
            
            # If we converged to a very small value, we can stop or continue to polish
            if best_f < 1e-6:
                # We found the minimum. Use remaining budget to polish if needed, 
                # but AOCC is already maxed out for this instance effectively.
                # Actually, AOCC is based on the trajectory. 
                # Once we hit the min, n_i becomes 0 (or close). 
                # So we should stop wasting calls if we are at the min?
                # No, the evaluator records every call. 
                # If we stop, the last n_i is repeated. 
                # So it's fine to stop.
                break

        return best_x, best_f
