"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
Adaptive Coordinate Descent for anisotropic refinement.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        # 64 points to ensure we find the basin of attraction
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return None, np.inf
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Stochastic Line Search with Adaptive Step Sizes ---
        # Bent Cigar is highly anisotropic. We use a line-search approach:
        # 1. Sample random directions (including coordinate axes).
        # 2. For each direction, perform a golden-section search (or similar) 
    #    to find the local minimum along that line.
        # 3. Update the current point to the best found.
        # This leverages the deterministic nature of 1D optimization to rapidly
        # align with the narrow valley.
        
        # Initialize step sizes (search radius) for the line search.
        # We keep a global radius and adapt it.
        sigma = 1.0
        rng = np.random.default_rng(12345)
        
        current_x = x0.copy()
        current_fx = best_fx
        
        # Pre-allocate direction cache? No, just generate on the fly.
        # We will cycle through a set of directions.
        # To ensure we cover the anisotropy, we always include coordinate axes.
        
        # Stagnation tracking
        no_improve_count = 0
        max_no_improve = 10
        
        # Golden Section Search parameters
        gr = (np.sqrt(5) - 1) / 2  # ~0.618
        
        def line_search(x_center, direction, radius, current_fx):
            """
            Perform a 1D search along x_center + t * direction for t in [-radius, radius].
            Returns best_t, best_fx.
            """
            # If radius is very small, just try a few points
            if radius < 1e-6:
                return 0.0, current_fx
            
            # Golden section search bounds
            a = -radius
            b = radius
            
            # Initial points
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate initial points
            x_c = x_center + c * direction
            x_c = np.clip(x_c, -5, 5)
            x_d = x_center + d * direction
            x_d = np.clip(x_d, -5, 5)
            
            try:
                fc = wrapped_obj(x_c)
                fd = wrapped_obj(x_d)
            except Exception:
                return 0.0, current_fx
            
            # Also check center (t=0) to ensure we don't miss the current best if it's better
            # Actually, current_fx is the value at x_center.
            # If fc or fd is worse than current_fx, the minimum might be at 0 or we need to narrow.
            # Standard GS assumes unimodal. Bent Cigar is unimodal along any line through the minimum.
            # But we might be far away.
            
            # Simple refinement: iterate GS for a fixed number of steps or until interval is small
            # Max 10 iterations to keep cost low (2 evals per iter)
            for _ in range(10):
                if time.time() >= deadline or calls_used >= budget:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    x_c = x_center + c * direction
                    x_c = np.clip(x_c, -5, 5)
                    try:
                        fc = wrapped_obj(x_c)
                    except Exception:
                        break
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    x_d = x_center + d * direction
                    x_d = np.clip(x_d, -5, 5)
                    try:
                        fd = wrapped_obj(x_d)
                    except Exception:
                        break
                
                if abs(b - a) < 1e-5:
                    break
            
            # The best point is likely between a and b.
            # Check the midpoint or the best of c, d?
            # c and d are the last evaluated points.
            # Let's just return the best of c, d, and the center (which we already know).
            # Actually, we should return the t corresponding to the best fx found in this search.
            
            # We have evaluated x_c (fc) and x_d (fd).
            # The current center is x_center (current_fx).
            # We need to find the t that gave the min.
            # This is tricky because we didn't store all t's.
            # Let's just evaluate the midpoint of the final interval [a, b] to get a good estimate.
            # Or better, just return the best of the last two evaluated points.
            
            # To be safe and simple, let's just pick the best of c, d, and 0.
            # But we need the t value.
            # Let's restart with a simpler approach:
            # 1. Evaluate at -radius, 0, radius.
            # 2. Pick best interval.
            # 3. Do a few bisection/gs steps.
            
            # Let's stick to the GS result.
            # If fc < fd, min is in [a, d]. Else [c, b].
            # The best value is min(fc, fd, current_fx).
            # We need the x corresponding to the best.
            
            # This implementation is getting complex to track the exact t.
            # Let's simplify:
            # We will just do a few random steps along the direction and pick the best.
            # This is less efficient than GS but simpler and robust.
            
            # Actually, let's use the GS points c and d.
            # If fc < fd, the best is likely closer to c.
            # Let's just evaluate the center of the final interval [a, b].
            t_mid = (a + b) / 2.0
            x_mid = x_center + t_mid * direction
            x_mid = np.clip(x_mid, -5, 5)
            try:
                f_mid = wrapped_obj(x_mid)
            except Exception:
                f_mid = np.inf
            
            # Compare f_mid, fc, fd, current_fx
            best_f = current_fx
            best_t = 0.0
            
            if fc < best_f:
                best_f = fc
                best_t = c
            if fd < best_f:
                best_f = fd
                best_t = d
            if f_mid < best_f:
                best_f = f_mid
                best_t = t_mid
                
            return best_t, best_f

        # Main Loop
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            improved = False
            
            # Generate a set of directions to try
            # 1. Coordinate axes (5 directions)
            # 2. Random directions (5 directions)
            directions = []
            for i in range(dim):
                d_i = np.zeros(dim)
                d_i[i] = 1.0
                directions.append(d_i)
            
            for _ in range(dim):
                d_r = rng.standard_normal(dim)
                norm = np.linalg.norm(d_r)
                if norm > 0:
                    d_r /= norm
                directions.append(d_r)
                
            # Shuffle directions to avoid bias
            idx = rng.permutation(len(directions))
            
            for i in idx:
                if time.time() >= deadline or calls_used >= budget:
                    break
                
                direction = directions[i]
                
                # Perform line search
                best_t, best_f = line_search(current_x, direction, sigma, current_fx)
                
                if best_f < current_fx:
                    current_x = current_x + best_t * direction
                    current_x = np.clip(current_x, -5, 5)
                    current_fx = best_f
                    improved = True
                    # Success: Expand sigma slightly
                    sigma *= 1.2
                    sigma = min(sigma, 5.0)
                else:
                    # Failure: Shrink sigma
                    sigma *= 0.8
                    if sigma < 1e-6:
                        sigma = 1e-6
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Stuck?
                    if sigma < 1e-4:
                        break
                    # Try to kick out
                    sigma *= 2.0
                    no_improve_count = 0

        # Final update of best if line search found a better point
        if current_fx < best_fx:
            best_fx = current_fx
            best_x = current_x

        return best_x, best_fx
