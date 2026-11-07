"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + axis-aligned coordinate-wise line search
    for Schaffer's F7 (ill-conditioned, axis-aligned structure)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 280.0  # 20s safety margin before 300s wall-clock

        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                val = float(func(x))
            except Exception:
                return None
            calls[0] += 1
            if not np.isfinite(val):
                return None
            return val

        def time_ok():
            return time.time() < DEADLINE

        # ---- Phase 1: Deterministic Multi-Resolution Grid Search ----
        # We start with a coarse grid to find the approximate basin,
        # then refine the grid around the best point.
        
        # Initial coarse grid: 3 points per dimension -> 3^5 = 243 points
        # This is well within the 500 budget.
        grid_res = 3
        points = np.array(np.meshgrid(*[np.linspace(-5.0, 5.0, grid_res)] * dim)).T.reshape(-1, dim)
        
        best_x = None
        best_f = float('inf')
        
        for p in points:
            if not time_ok() or calls[0] >= budget:
                break
            fp = safe_call(p)
            if fp is not None and fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        if best_x is None:
            # Fallback if even the first call fails or budget is 0
            return np.zeros(dim), 0.0

        # Refine grid around the best point found
        # We will do 2 more levels of refinement if budget allows.
        # Level 2: 5 points per dimension in a local box around best_x
        # Level 3: 5 points per dimension in an even tighter box
        
        current_best = best_x.copy()
        current_best_f = best_f
        
        # Define refinement levels: (resolution, radius)
        # Radius is half-width of the box around current_best
        # We want to stay within [-5, 5]
        levels = [
            (5, 2.0),  # Box of size 4x4x4x4x4, 5^5 = 3125 points (too many, we will limit)
            (5, 0.5),  # Box of size 1x1x1x1x1
            (5, 0.1),  # Box of size 0.2x0.2x0.2x0.2x0.2
        ]
        
        for res, radius in levels:
            if calls[0] >= budget or not time_ok():
                break
            
            # Generate grid points in the local box
            # To avoid generating too many points (e.g. 3125), we check budget
            # 5^5 = 3125. If we have ~250 calls left, we can't do a full 5^5 grid.
            # So we adapt the resolution or just pick a subset?
            # Better: reduce resolution if budget is tight.
            # Let's just use a smaller grid if budget is low.
            # For radius 2.0, 5^5 is too big if we only have 250 calls.
            # Let's use 3^5 = 243 for the first refinement if budget < 300, else 5^5.
            
            if calls[0] + (res ** dim) > budget:
                # Reduce resolution to fit budget
                # Try res=3
                if calls[0] + (3 ** dim) <= budget:
                    res = 3
                else:
                    # Try res=2 (just corners/midpoints)
                    if calls[0] + (2 ** dim) <= budget:
                        res = 2
                    else:
                        break # Not enough budget for any grid refinement
            
            # Generate points
            # Use linspace within [current_best - radius, current_best + radius]
            # Clip to [-5, 5]
            low = np.clip(current_best - radius, -5.0, 5.0)
            high = np.clip(current_best + radius, -5.0, 5.0)
            
            # If low == high for any dim, skip or handle carefully
            if np.allclose(low, high):
                break
                
            points = np.array(np.meshgrid(*[np.linspace(low[i], high[i], res) for i in range(dim)])).T.reshape(-1, dim)
            
            local_best_x = current_best
            local_best_f = current_best_f
            
            for p in points:
                if not time_ok() or calls[0] >= budget:
                    break
                fp = safe_call(p)
                if fp is not None and fp < local_best_f:
                    local_best_f = fp
                    local_best_x = p.copy()
                    
            if local_best_f < current_best_f:
                current_best = local_best_x
                current_best_f = local_best_f
            else:
                # No improvement in this refinement level, break to save time for line search
                break

        x_best = current_best
        f_best = current_best_f

        # ---- Phase 2: Coordinate-wise Line Search ----
        # Schaffer's F7 is ill-conditioned and axis-aligned.
        # We perform a coordinate-wise search to fine-tune the solution.
        # For each coordinate, we perform a 1D search (golden section or ternary)
        # to minimize the function along that axis, holding others constant.
        
        # We will iterate this process a few times or until budget/time runs out.
        # Since 1D search is cheap, we can do many iterations.
        
        # Define a 1D minimization function using Golden Section Search
        def golden_section_search(func_1d, a, b, tol=1e-6, max_iter=100):
            """
            Minimize func_1d in [a, b] using Golden Section Search.
            func_1d takes a scalar x and returns a float.
            """
            gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate ~ 0.618
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = func_1d(c)
            fd = func_1d(d)
            
            if fc is None or fd is None:
                return (a + b) / 2, min(fc if fc is not None else float('inf'), fd if fd is not None else float('inf'))
                
            for _ in range(max_iter):
                if not time_ok() or calls[0] >= budget:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = func_1d(c)
                    if fc is None:
                        break
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = func_1d(d)
                    if fd is None:
                        break
                if abs(b - a) < tol:
                    break
            
            x_opt = (a + b) / 2
            f_opt = func_1d(x_opt)
            if f_opt is None:
                f_opt = min(fc, fd) if fc is not None and fd is not None else float('inf')
            return x_opt, f_opt

        # Iterative Coordinate Descent
        # We'll do several passes over the coordinates.
        # Each pass involves 1D search for each coordinate.
        # 1D search uses ~100 function calls.
        # 5 coords * 100 calls = 500 calls per pass.
        # So we can likely only do 1 pass if we start with ~500 budget left.
        # But we used ~300 for grid. So we have ~200 left.
        # 1D search max_iter=100 is too high. Let's reduce to 30-40 iterations.
        # 5 * 40 = 200 calls. This fits perfectly.
        
        max_1d_iter = 40
        num_passes = 3 # We'll try up to 3 passes, but break if budget runs out
        
        for _pass in range(num_passes):
            if not time_ok() or calls[0] >= budget:
                break
            
            x_curr = x_best.copy()
            f_curr = f_best
            
            for i in range(dim):
                if not time_ok() or calls[0] >= budget:
                    break
                
                # Define 1D function
                def func_1d(x_val, idx=i, x_base=x_curr):
                    x_test = x_base.copy()
                    x_test[idx] = x_val
                    return safe_call(x_test)
                
                # Search range: [x_curr[i] - 1.0, x_curr[i] + 1.0]
                # Clip to [-5, 5]
                a = max(-5.0, x_curr[i] - 1.0)
                b = min(5.0, x_curr[i] + 1.0)
                
                if a >= b:
                    continue
                    
                x_new_i, f_new_i = golden_section_search(func_1d, a, b, tol=1e-5, max_iter=max_1d_iter)
                
                if f_new_i is not None and f_new_i < f_curr:
                    x_curr[i] = x_new_i
                    f_curr = f_new_i
                    x_best = x_curr.copy()
                    f_best = f_curr
        
        return x_best, f_best
# EVOLVE-BLOCK-END
