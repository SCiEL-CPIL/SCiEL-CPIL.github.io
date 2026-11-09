import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin

        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.full(n, np.nan)
        best_f = np.inf
        all_evaluated = [] # Store (f, x) to pick diverse starts later if needed

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Strategic Global Sampling ---
        # We need to find a few good basins. Rastrigin has many local minima.
        # Use a stratified sampling approach to cover the space efficiently.
        
        # 1. Deterministic points: Center and corners of the sub-box [-4, 4] 
        # (since minimizer is in [-4, 4])
        candidates = []
        candidates.append(np.zeros(n))
        
        # Add points from a coarse grid in [-4, 4]
        # 5D space. A full grid is too expensive.
        # Let's use a Latin Hypercube or just random with a fixed seed for structure.
        # Actually, let's use a small set of "anchor" points that are well spread.
        
        # Generate 20-30 diverse points to identify top basins.
        # We will spend ~20% of budget on global search.
        global_budget = int(total_budget * 0.2)
        
        # Create a set of diverse starting points
        # 1. Origin
        # 2. Random uniform points
        rng = np.random.default_rng(42)
        
        # Add some structured points: vertices of a hypercube scaled to [-4, 4]
        # But 2^5 = 32 points. That's a good number.
        # Let's add the 32 corners of the [-4, 4] box.
        corners = []
        for i in range(2**n):
            p = np.zeros(n)
            for j in range(n):
                if (i >> j) & 1:
                    p[j] = 4.0
                else:
                    p[j] = -4.0
            corners.append(p)
            
        # Add center
        candidates.extend(corners)
        candidates.append(np.zeros(n))
        
        # If we have budget for more, add random points
        if global_budget > len(candidates):
            num_rand = global_budget - len(candidates)
            random_points = rng.uniform(-4, 4, (num_rand, n))
            candidates.extend(random_points)
        else:
            # If we can't afford all corners, subsample them or just use random
            # But 32 is small. Let's just evaluate the first global_budget points.
            pass

        # Evaluate global candidates
        # We want to keep track of the best few to use as restart points.
        # Let's store the top K results from this phase.
        top_k = 5
        local_best = []
        
        for p in candidates:
            if not time_ok() or calls_made >= global_budget:
                break
            val = do_call(p)
            # Insert into local_best if it's good
            if val < np.inf:
                local_best.append((val, p.copy()))
                if len(local_best) > top_k:
                    local_best.sort(key=lambda x: x[0])
                    local_best = local_best[:top_k]

        # --- Phase 2: Multi-start Local Optimization (CMA-ES) ---
        # We will use the remaining budget to run CMA-ES from the best starting points.
        # CMA-ES is better than NM for rotated functions.
        
        remaining_budget = total_budget - calls_made
        
        # If we didn't find any valid points (shouldn't happen), fallback to random
        if not local_best:
             if calls_made < total_budget:
                 do_call(np.zeros(n))
             return best_x, best_f

        # We have up to 5 restarts.
        # Allocate budget to each restart.
        # If we have 400 calls left, and 5 restarts, 80 calls each.
        # CMA-ES in 5D usually needs ~300-500 evaluations for high precision, but 
        # even 50-100 can significantly reduce the error if started in the basin.
        
        # Strategy: 
        # 1. Run CMA-ES on the global best point with a larger budget.
        # 2. Run CMA-ES on the 2nd, 3rd best points with smaller budgets to see if they are actually better basins.
        
        # Sort local_best by fitness
        local_best.sort(key=lambda x: x[0])
        
        # Define a wrapper for CMA-ES that respects budget and time
        def cma_func_wrapper(x):
            if calls_made >= total_budget or not time_ok():
                return best_f # Return current best to stop optimization gently? 
                              # Actually, returning a constant might confuse CMA-ES.
                              # It's better to raise an exception or just stop.
                              # scipy minimize handles exceptions by stopping.
                              # But we want to use up the budget if possible.
                              # Let's just return a large value to discourage further exploration?
                              # No, if we return best_f, it might think it converged.
                              # Let's just return the actual value if we have budget, 
                              # but if we hit the limit, we should stop.
                              # The simplest is to check budget inside and return a very large number 
                              # if we are out, which will likely cause CMA-ES to stop or 
                              # we can catch an exception.
                              # Better: just let it call do_call which returns inf if out of budget.
                              # But do_call returns inf. CMA-ES might handle inf poorly.
                              # Let's wrap do_call to return a large finite number if out of budget.
            
            if calls_made >= total_budget:
                return 1e10
            if not time_ok():
                return 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # We need to be careful with nonlocal variables in nested functions in Python 3.
        # The above cma_func_wrapper uses nonlocal implicitly? No, it reads best_f, best_x, calls_made.
        # It needs to update them. So we must use nonlocal.
        
        def cma_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return 1e10
            if not time_ok():
                return 1e10
            
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Restart 1: Best point
        # Use 60% of remaining budget
        if remaining_budget > 20:
            b1 = int(remaining_budget * 0.6)
            start_x = local_best[0][1]
            
            # CMA-ES options
            # sigma0: initial step size. For Rastrigin, 1.0 is reasonable.
            # maxiter: limit iterations, but we control via budget.
            # tol: small tolerance.
            
            try:
                res = minimize(cma_func, start_x, method='CMA', 
                               options={
                                   'maxiter': b1, # maxiter is iterations, not function calls.
                                            # CMA-ES calls ~ (pop_size + 1) * maxiter times?
                                            # pop_size default is 4 + floor(3 ln n) = 4 + 3*ln(5) ~ 9.
                                            # So 10 calls per iter.
                                            # If b1=240, maxiter should be ~24.
                                   'maxiter': max(10, b1 // 10),
                                   'sigma0': 1.0,
                                   'tolx': 1e-6,
                                   'tolfun': 1e-6,
                                   'verbose': False,
                                   'bounds': [(-5.0, 5.0)] * n
                               })
            except Exception:
                pass

        # Restart 2: 2nd best point
        remaining_budget = total_budget - calls_made
        if len(local_best) > 1 and remaining_budget > 20:
            b2 = int(remaining_budget * 0.4)
            start_x = local_best[1][1]
            try:
                res = minimize(cma_func, start_x, method='CMA', 
                               options={
                                   'maxiter': max(10, b2 // 10),
                                   'sigma0': 1.0,
                                   'tolx': 1e-6,
                                   'tolfun': 1e-6,
                                   'verbose': False,
                                   'bounds': [(-5.0, 5.0)] * n
                               })
            except Exception:
                pass

        # Restart 3: 3rd best point if budget remains
        remaining_budget = total_budget - calls_made
        if len(local_best) > 2 and remaining_budget > 10:
            b3 = int(remaining_budget * 0.5)
            start_x = local_best[2][1]
            try:
                res = minimize(cma_func, start_x, method='CMA', 
                               options={
                                   'maxiter': max(5, b3 // 10),
                                   'sigma0': 0.5, # Smaller step for later restarts
                                   'tolx': 1e-6,
                                   'tolfun': 1e-6,
                                   'verbose': False,
                                   'bounds': [(-5.0, 5.0)] * n
                               })
            except Exception:
                pass

        # Final Polish: If we have very little budget left, do a few random steps around best_x?
        # Or just return.
        
        return best_x, best_f
