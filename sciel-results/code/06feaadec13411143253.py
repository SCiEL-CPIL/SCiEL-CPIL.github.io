"""
Optimizer for BBOB f12 (Bent Cigar) using Coordinate Descent with Line Search
and Nelder-Mead polishing.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        # Safety margin: evaluator kills at 330s. We target 270s to be safe.
        deadline = t0 + 270.0 
        
        calls = [0]
        best_val = float('inf')
        best_x = None

        def f(x):
            nonlocal best_val, best_x
            calls[0] += 1
            try:
                val = float(func(x))
            except Exception:
                return 1e18
            if not np.isfinite(val):
                return 1e18
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        def remaining_time():
            return deadline - time.time()

        def line_search_1d(x, i, fx_current, initial_step=1.0, max_iter=50):
            """
            Perform 1D line search along dimension i.
            Uses a combination of parabolic interpolation and bisection.
            Returns the best x and fx found.
            """
            if calls[0] >= budget or remaining_time() < 0.1:
                return x, fx_current

            # Initial direction: try +/- step
            # We need to find a step size that decreases the function.
            # Start with a moderate step.
            
            x_curr = x.copy()
            fx_curr = fx_current
            
            # Determine direction and initial step size
            # Try positive first
            step = initial_step
            # If function increases, try negative or smaller step
            # Simple strategy: binary search for the minimizer in a bracket.
            
            # Bracketing phase
            # Try to find a bracket [lo, hi] where the function is lower at the middle
            # or just find a decreasing step.
            
            # Let's use a more robust approach: 
            # 1. Find a step that decreases f.
            # 2. Refine using parabolic interpolation.
            
            # Try steps: 1, 0.5, 0.25, ... down to 1e-4
            # Or expand: 1, 2, 4... if we want to escape flat regions, but we are minimizing.
            
            # Since it's convex-ish, we just need to find the minimum along the line.
            # Let's use a simple bisection on the derivative if we had it, but we don't.
            # We will use ternary search or parabolic fit on 3 points.
            
            # Points: x - step, x, x + step
            # Actually, simpler: just do a golden section or parabolic search on a range.
            
            # Let's define a range around x[i].
            # The scale of the problem is [-5, 5].
            # If we are far from min, we need large steps. If close, small steps.
            # Adaptive step: based on previous success? 
            # For simplicity and robustness, we'll use a fixed set of probe points 
            # that expand if no improvement is found, or contract if we are oscillating.
            
            # However, for Bent Cigar, the step size required varies wildly by dimension.
            # Dimension 0 (usually) needs step ~ 1000 (or clipped to bounds).
            # Other dims need step ~ 1e-4.
            
            # Strategy: 
            # 1. Try a "large" step (1.0) and "small" step (1e-3).
            # 2. Pick the better one.
            # 3. Refine around the better one.
            
            candidates = []
            # Probe points relative to x[i]
            # We will evaluate x with x[i] modified.
            
            def eval_mod(delta):
                if calls[0] >= budget or remaining_time() < 0.05:
                    return None, None
                x_new = x_curr.copy()
                x_new[i] += delta
                # Clip to bounds [-5, 5]
                x_new[i] = np.clip(x_new[i], -5.0, 5.0)
                if x_new[i] == x_curr[i]:
                    return x_new, fx_curr
                val = f(x_new)
                return x_new, val

            # Initial probes
            # Large step to catch the wide valley
            x_l, fx_l = eval_mod(1.0)
            x_s, fx_s = eval_mod(1e-3)
            
            # If no improvement from 1.0, try -1.0? 
            # Actually, just pick the best of the two and refine.
            # But we might be on the wrong side.
            # Let's check x - 1.0 and x - 1e-3 if neither improved.
            
            if fx_l > fx_curr and fx_s > fx_curr:
                 x_l, fx_l = eval_mod(-1.0)
                 x_s, fx_s = eval_mod(-1e-3)
                 
            # Select best candidate
            if fx_l < fx_s:
                x_best = x_l
                fx_best = fx_l
                # Refine around x_l
                # Use parabolic interpolation with x_curr and x_l? 
                # We need a third point.
                # Let's just do a binary search between x_curr[i] and x_l[i]
                # assuming convexity along this axis.
                lo = x_curr[i]
                hi = x_l[i]
                fx_lo = fx_curr
                fx_hi = fx_l
            else:
                x_best = x_s
                fx_best = fx_s
                lo = x_curr[i]
                hi = x_s[i]
                fx_lo = fx_curr
                fx_hi = fx_s
                
            # If no improvement at all, return current
            if fx_best >= fx_curr:
                return x_curr, fx_curr

            # Binary search for minimum in [lo, hi]
            # We assume the function is unimodal in this interval.
            for _ in range(20): # Max 20 iterations for refinement
                if calls[0] >= budget or remaining_time() < 0.05:
                    break
                if hi - lo < 1e-6:
                    break
                mid = (lo + hi) / 2.0
                x_mid = x_curr.copy()
                x_mid[i] = mid
                x_mid[i] = np.clip(x_mid[i], -5.0, 5.0)
                if x_mid[i] == x_curr[i]:
                    break
                fx_mid = f(x_mid)
                
                if fx_mid < fx_lo:
                    hi = mid
                    fx_hi = fx_mid
                    if fx_mid < fx_best:
                        fx_best = fx_mid
                        x_best = x_mid.copy()
                else:
                    lo = mid
                    fx_lo = fx_mid
                    # If fx_mid is better than current best (which might be fx_hi if we moved hi?)
                    # Actually, we are tracking fx_best separately.
                    if fx_mid < fx_best:
                        fx_best = fx_mid
                        x_best = x_mid.copy()
                        
            return x_best, fx_best

        def coordinate_descent(x0, fx0, max_iter=1000):
            x = x0.copy()
            fx = fx0
            # Cycle through dimensions
            # For Bent Cigar, order matters less if we cycle, but 
            # the "wide" dimension usually converges in 1-2 steps.
            
            for _ in range(max_iter):
                if calls[0] >= budget or remaining_time() < 1.0:
                    break
                
                prev_fx = fx
                improved = False
                for i in range(dim):
                    if calls[0] >= budget or remaining_time() < 0.5:
                        break
                    x_new, fx_new = line_search_1d(x, i, fx)
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                    # If no improvement, we might be at a local min or noise.
                    # For robustness, if we cycle through all dims without improvement, stop.
                
                if not improved:
                    break
            return x, fx

        def nelder_mead(x0, fx0, max_iter=100):
            """
            Standard Nelder-Mead implementation for final polishing.
            """
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            
            # Add other vertices
            # Step size for initialization
            # Use a small step for polish
            step = 0.1 
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += step
                x_i[i] = np.clip(x_i[i], -5.0, 5.0)
                f_i = f(x_i)
                simplex.append(x_i)
                fvals.append(f_i)
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if calls[0] >= budget or remaining_time() < 0.5:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = f(x_r)
                
                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = f(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = f(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = f(simplex[i])
            
            # Return best
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Main Loop
        # 1. Multi-start
        rng = np.random.default_rng(42)
        
        # Generate starts
        starts = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.array([4.0] * dim),
            np.array([-4.0] * dim)
        ]
        
        # We have a budget. 
        # Coordinate descent is cheap (1D searches). 
        # We will run CD on all starts, then NM on the best one.
        
        # Phase 1: Coordinate Descent on all starts
        # Limit iterations per start to ensure we finish.
        # Each CD step costs ~dim * (1 + 20) calls? No, line_search_1d does ~2 probes + 20 binary steps = ~22 calls per dim.
        # 5 dims * 22 = 110 calls per CD iteration.
        # If we allow 5 CD iterations per start, that's 550 calls per start. Too much.
        # We need to be more efficient.
        
        # Revised Line Search: 
        # The binary search loop of 20 is expensive. 
        # For Bent Cigar, we can often find the min with fewer iterations if we use parabolic fit.
        # Or just limit the binary search iterations to 5-10.
        
        # Let's reduce the binary search iterations in line_search_1d to 8.
        # And limit CD iterations to 2 per start.
        # 2 iter * 5 dim * (2 + 8) calls = 100 calls per start.
        # 5 starts * 100 = 500 calls. Exactly budget.
        # This is risky.
        
        # Better strategy:
        # 1. Run CD for 1 iteration on all starts (cheap, finds basin).
        # 2. Pick the best start.
        # 3. Run CD for more iterations on the best start.
        # 4. Run NM on the best result.
        
        # Phase 1: Quick CD (1 iter) on all starts
        best_x_cd = None
        best_fx_cd = float('inf')
        
        for s in starts:
            if calls[0] >= budget or remaining_time() < 10.0:
                break
            s = np.clip(s, -5.0, 5.0)
            fx = f(s)
            # 1 iteration of CD
            x_cd, fx_cd = coordinate_descent(s, fx, max_iter=1)
            if fx_cd < best_fx_cd:
                best_fx_cd = fx_cd
                best_x_cd = x_cd.copy()
        
        if best_x_cd is None:
             # Fallback
             best_x_cd = starts[0]
             best_fx_cd = f(best_x_cd)

        # Phase 2: Deep CD on the best start
        # We have used ~5 * (1 + 5*10) = 255 calls approx.
        # Remaining: ~245 calls.
        # Run CD for ~2 more iterations.
        x_cd2, fx_cd2 = coordinate_descent(best_x_cd, best_fx_cd, max_iter=2)
        
        # Phase 3: Nelder-Mead Polish
        # NM uses ~ n+1 calls per iteration. 6 calls/iter.
        # We can afford ~30-40 iterations.
        x_final, fx_final = nelder_mead(x_cd2, fx_cd2, max_iter=40)
        
        # Return the best found so far
        return best_x, best_val

# EVOLVE-BLOCK-END
