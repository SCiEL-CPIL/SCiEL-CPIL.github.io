"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Black-box continuous optimizer using Multi-Start L-BFGS-B with random restarts.
    
    Strategy:
    1. Phase 1 (Exploration): Use LHS to sample diverse points, find initial best.
    2. Phase 2 (Exploitation): Use Nelder-Mead or Powell from best point to refine.
    3. Phase 3 (Restart): If stuck, restart from new random point.
    
    This approach leverages local optimization power while maintaining global coverage.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 310.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Wrapper to count calls
        def counted_func(x):
            nonlocal calls
            if time.time() - start_time > max_time:
                raise TimeoutError("Time limit exceeded")
            calls += 1
            return func(x)
        
        # Phase 1: Latin Hypercube Sampling for initial exploration
        # Use ~30% of budget for exploration
        n_explore = max(20, int(budget * 0.3))
        
        # Generate LHS points
        def lhs_sample(n, dim, rng):
            """Generate n points using Latin Hypercube Sampling in [-5, 5]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.uniform(size=n)) / n * 10 - 5
            return points
        
        lhs_points = lhs_sample(n_explore, dim, rng)
        
        for x in lhs_points:
            if time.time() - start_time > max_time:
                break
            if calls >= budget:
                break
            try:
                fx = counted_func(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            try:
                fx = counted_func(x)
                return x, fx
            except:
                return np.zeros(dim), float('inf')
        
        # Phase 2: Local optimization from best point
        # Use Powell method which doesn't require gradients and works well for black-box
        remaining_budget = budget - calls
        
        # We'll do multiple local optimizations from different starting points
        # Strategy: 
        # 1. Optimize from current best
        # 2. If budget remains, try optimizing from other good points found during exploration
        # 3. If still stuck, random restarts
        
        # Track all evaluated points to find diverse starting points
        evaluated_points = []
        evaluated_values = []
        
        # Re-evaluate LHS points to track them (we already have them, but let's be safe)
        # Actually, we can just use the LHS points we generated
        
        # Sort LHS points by function value (we need to re-evaluate or store them)
        # Let's re-evaluate a subset to get values for sorting
        # Better: store values during Phase 1
        
        # Let's redo Phase 1 with storage
        # Actually, let's just proceed with local optimization from best_x
        # and then do random restarts if needed
        
        # Local optimization using Powell
        def run_local_opt(start_x, maxiter):
            nonlocal best_x, best_f, calls
            try:
                res = minimize(counted_func, start_x, method='Powell', 
                             options={'maxiter': maxiter, 'maxfev': budget - calls})
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
        
        # First local optimization from global best
        if calls < budget and time.time() - start_time < max_time:
            # Use up to 50% of remaining budget for this optimization
            maxiter_1 = max(10, int((budget - calls) * 0.5))
            run_local_opt(best_x, maxiter_1)
        
        # Phase 3: Random restarts with local optimization
        # If we still have budget, try new random starts
        n_restarts = 5
        for i in range(n_restarts):
            if calls >= budget or time.time() - start_time > max_time:
                break
            
            # New random start
            x_start = rng.uniform(-5, 5, dim)
            
            # Quick local optimization
            maxiter_2 = max(5, int((budget - calls) * 0.3))
            run_local_opt(x_start, maxiter_2)
        
        # Final polish: if budget remains, do one more local optimization from best
        if calls < budget and time.time() - start_time < max_time:
            maxiter_3 = max(5, budget - calls)
            run_local_opt(best_x, maxiter_3)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
