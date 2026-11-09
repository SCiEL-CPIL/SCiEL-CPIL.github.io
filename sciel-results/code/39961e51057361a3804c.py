"""Rewritten optimizer: Multi-start Coordinate Descent with Golden Section Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizes BBOB f3 (Rastrigin) using multi-start coordinate descent.
    Uses Golden Section Search for 1D optimization along each axis.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def golden_section_search(f, a, b, x_fixed, dim_index, tol=1e-8, max_iter=50):
            """
            1D Golden Section Search along dim_index.
            x_fixed is the vector with all coords fixed except dim_index.
            Returns the optimal coordinate value.
            """
            if calls_made >= budget or time.time() > deadline:
                return x_fixed[dim_index]
            
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            x_c = x_fixed.copy()
            x_c[dim_index] = c
            x_d = x_fixed.copy()
            x_d[dim_index] = d
            
            f_c = safe_call(x_c)
            f_d = safe_call(x_d)
            
            if calls_made >= budget or time.time() > deadline:
                return c if f_c < f_d else d

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break
                if b - a < tol:
                    break
                    
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c = x_fixed.copy()
                    x_c[dim_index] = c
                    f_c = safe_call(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_d = x_fixed.copy()
                    x_d[dim_index] = d
                    f_d = safe_call(x_d)
                    
                if calls_made >= budget or time.time() > deadline:
                    break

            # Return the midpoint of the final interval
            return (a + b) / 2.0

        def coordinate_descent(x0, max_rounds=5):
            """
            Perform coordinate descent starting from x0.
            """
            x = x0.copy()
            fx = safe_call(x)
            
            for _ in range(max_rounds):
                if calls_made >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Define search window around current x[i]
                    # Use a window that covers the local minimum.
                    # Rastrigin period is 2.0. A window of 1.0 is usually safe if we are close.
                    # Start with a wider window to ensure we capture the minimum if we are far.
                    # However, if we are far, we might jump to a wrong local min.
                    # Strategy: Start with a window of +/- 1.0. If no improvement, try +/- 2.0?
                    # For efficiency, let's stick to +/- 1.0 first. If we start well, this is enough.
                    
                    lo = max(-5.0, x[i] - 1.0)
                    hi = min(5.0, x[i] + 1.0)
                    
                    # If the window is too small, skip or adjust
                    if hi - lo < 1e-6:
                        continue

                    new_coord = golden_section_search(func, lo, hi, x, i)
                    
                    # Check if the new point is better
                    x_new = x.copy()
                    x_new[i] = new_coord
                    f_new = safe_call(x_new)
                    
                    if f_new < fx - 1e-10:
                        x = x_new
                        fx = f_new
                        improved = True
                        # Update global best if applicable (safe_call does this)
                    else:
                        # If no improvement, we might be stuck or in a flat region.
                        # Optionally try a wider search if we haven't converged.
                        # For now, assume +/- 1.0 is sufficient for local convergence.
                        pass
                        
                if not improved:
                    break
            return x, fx

        # Phase 1: Generate diverse starting points
        # Use a combination of deterministic and random points
        # 1. Origin
        # 2. Corners of the box (scaled)
        # 3. Random points
        
        rng = np.random.default_rng(42)
        
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        
        # Mid-points of axes (positive and negative)
        for i in range(dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                pt[i] = sign * 2.0 # Rastrigin min is at 0, next at 2. 2.0 is a good jump.
                starts.append(pt)
        
        # Random points
        # We have ~500 budget. 
        # Each coordinate descent round takes ~dim * (2 + max_iter) calls.
        # With max_iter=20, that's ~5 * 22 = 110 calls per full round.
        # We can afford ~4-5 full rounds.
        # Let's do 5 starting points.
        for _ in range(4):
            pt = rng.uniform(-4.0, 4.0, dim)
            starts.append(pt)
        
        # Sort starts by some heuristic? Or just run them.
        # To maximize early progress (AOCC), we should run the most promising ones first.
        # Origin is usually good.
        
        current_best_x = np.zeros(dim)
        current_best_f = np.inf
        
        for start_pt in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Run coordinate descent from this start
            # Limit the number of rounds per start to ensure we visit multiple starts
            # If we use all budget on one, we might miss a better basin.
            # But if the first one is good, we should refine it.
            # Strategy: Run 2 rounds for each start. Then take the best and run more?
            
            x_res, f_res = coordinate_descent(start_pt, max_rounds=2)
            
            if f_res < current_best_f:
                current_best_f = f_res
                current_best_x = x_res.copy()
        
        # Phase 2: Refine the best point found
        if calls_made < budget and time.time() < deadline:
            # Run more coordinate descent rounds on the best point
            # Use tighter tolerance or more rounds
            x_res, f_res = coordinate_descent(current_best_x, max_rounds=5)
            if f_res < current_best_f:
                current_best_f = f_res
                current_best_x = x_res.copy()
                
        # Phase 3: Final polish with Nelder-Mead if budget remains
        if calls_made < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                # Estimate remaining budget
                remaining = budget - calls_made
                if remaining > 10:
                    result = minimize(
                        func,
                        x0=current_best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining // 2, 100),
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if result.fun < current_best_f:
                        current_best_f = result.fun
                        current_best_x = result.x
            except Exception:
                pass

        return current_best_x, current_best_f

# EVOLVE-BLOCK-END
