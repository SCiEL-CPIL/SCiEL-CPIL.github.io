import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f24 (Lunacek bi-Rastrigin).
    Strategy:
    1. Global exploration via Latin Hypercube Sampling (LHS) to identify basins.
    2. Local exploitation via L-BFGS-B (quasi-Newton) for fast convergence to minima.
    3. Final polishing with Nelder-Mead or coordinate descent if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: stop evaluating 5 calls before limit to ensure no exception
        # due to race conditions or overhead, though we track strictly.
        max_calls = budget - 5
        deadline = time.time() + 280  # 280s out of 300s limit
        
        global_best = None
        global_fx = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Fixed seed for reproducibility of the search strategy itself
        # The instances have hidden seeds, but our internal search randomness can be fixed.
        
        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            if calls_made >= max_calls:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        # Phase 1: Global Exploration using LHS
        # Lunacek has two main minima. We need enough points to likely hit both basins.
        # In 5D, 100 points is a good balance.
        n_lhs = 100
        lhs_points = []
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            points_d = (perm + 0.5) / n_lhs * 10.0 - 5.0
            lhs_points.append(points_d)
        lhs_points = np.column_stack(lhs_points)
        
        # Add a few strategic points: origin and some random ones just in case
        extra_points = [np.zeros(dim)]
        for _ in range(10):
            extra_points.append(rng.uniform(-4.5, 4.5, dim))
            
        all_starts = np.vstack([lhs_points, np.array(extra_points)])
        
        # Evaluate all starts
        # We will collect (fx, x) pairs
        candidates = []
        for x in all_starts:
            if calls_made >= max_calls or time.time() > deadline:
                break
            fx = evaluate(x)
            if fx < np.inf:
                candidates.append((fx, x.copy()))
                
        if not candidates:
            # Fallback if something went wrong
            x0 = np.zeros(dim)
            evaluate(x0)
            return global_best, global_fx

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Select top diverse candidates for local optimization
        # We want to optimize from the best few points to ensure we find the global min
        # if it's in a different basin than the current best.
        selected_starts = []
        min_distance = 1.5  # Threshold for considering points distinct enough
        
        # Always take the best one
        selected_starts.append(candidates[0][1])
        
        for fx, x in candidates[1:]:
            if len(selected_starts) >= 8:
                break
            # Check diversity
            is_diverse = True
            for sx in selected_starts:
                if np.linalg.norm(x - sx) < min_distance:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append(x)
                
        # If we didn't find enough diverse points, just take the next best ones
        while len(selected_starts) < 5:
            for fx, x in candidates:
                if x not in selected_starts:
                    selected_starts.append(x)
                    break
            if len(selected_starts) >= 5:
                break
            # Prevent infinite loop if candidates list is small
            if not candidates:
                break

        # Phase 2: Local Exploitation with L-BFGS-B
        bounds = [(-5.0, 5.0)] * dim
        
        for x0 in selected_starts:
            if calls_made >= max_calls or time.time() > deadline:
                break
            
            x0 = np.clip(x0, -5.0, 5.0)
            
            def objective(x):
                if calls_made >= max_calls or time.time() > deadline:
                    return np.inf
                return evaluate(x)
            
            try:
                # L-BFGS-B is efficient for smooth bounded problems
                # maxfev limits the number of function evaluations for this specific optimizer run
                # 200 is generous but capped by global budget
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': 100, # Don't let one run consume too much
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
                if result.fun < global_fx:
                    global_fx = result.fun
                    global_best = np.clip(result.x, -5.0, 5.0).copy()
                    
            except Exception:
                pass

        # Phase 3: Final Polishing
        # If we have budget left, do a high-precision polish on the global best.
        # Nelder-Mead is derivative-free and good for final steps, but can be slow.
        # Alternatively, a short run of L-BFGS-B with tighter tolerances.
        
        if global_best is not None and calls_made < max_calls - 10 and time.time() < deadline - 5:
            x0 = global_best.copy()
            
            def objective2(x):
                if calls_made >= max_calls or time.time() > deadline:
                    return np.inf
                return evaluate(x)
                
            try:
                # Use Nelder-Mead for a derivative-free final touch
                # It can sometimes escape shallow local minima or refine L-BFGS-B results
                result = minimize(
                    objective2,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds, # Nelder-Mead ignores bounds in scipy, but we clip in evaluate
                    options={
                        'maxiter': 50,
                        'maxfev': 50,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if result.fun < global_fx:
                    global_fx = result.fun
                    global_best = np.clip(result.x, -5.0, 5.0).copy()
                    
            except Exception:
                pass
        
        # Phase 4: Coordinate Descent / Random Walk Polish if budget remains
        remaining = max_calls - calls_made
        if remaining > 5 and global_best is not None and time.time() < deadline - 2:
            x = global_best.copy()
            fx = global_fx
            sigma = 0.01 # Small step size
            
            for _ in range(min(remaining - 2, 20)):
                if time.time() > deadline - 1:
                    break
                # Try random small perturbations
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 0.8 # Shrink step if successful
                else:
                    sigma *= 0.7 # Shrink step if failed
                    
        return global_best, global_fx
