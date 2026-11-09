import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy: Coarse-to-fine 1D grid search exploiting Rastrigin's separable structure
        # Rastrigin f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Each coordinate has local minima at integers. Global min is at some integer vector.
        # We can find the best integer for each coordinate independently, then refine.
        
        # Phase 1: Coarse grid search to identify the best integer basin for each dimension
        # The domain is [-5, 5], so integers are -5, -4, ..., 4, 5
        # But the shift might move the minimum. Let's use a finer grid first.
        
        # Actually, let's use a deterministic approach:
        # 1. Evaluate a coarse grid over the full domain to find promising regions
        # 2. For each dimension, find the best "integer-like" point
        # 3. Combine the best dimensions
        # 4. Refine with Nelder-Mead
        
        # Step 1: Coarse grid - evaluate at integer points and midpoints
        # This gives us 11 points per dimension (from -5 to 5 in steps of 1)
        # But we need to be efficient. Let's use a smaller grid first.
        
        # Alternative: Use the fact that Rastrigin has period 2*pi ~ 6.28
        # So within [-5, 5], there are about 2 periods per dimension.
        # Local minima are near integers.
        
        # Let's evaluate at all integer points first (11^5 = 161051 points - too many)
        # Instead, let's do coordinate-wise search.
        
        # Phase 1: Find the best integer for each coordinate
        # Start with x = 0
        x = np.zeros(n)
        fx = evaluate(x)
        
        # For each coordinate, try shifting by integers to find the best basin
        # We'll do this iteratively: for each coord, try -5 to 5, keep the best
        
        # But this is 11 * 5 = 55 evaluations just for the first pass
        # Then we can refine within each basin.
        
        # Let's do a more systematic approach:
        # 1. Evaluate at x=0
        # 2. For each dimension i, try x with x_i in {-4, -3, ..., 3, 4} (keeping others at 0)
        #    This is 9 * 5 = 45 evaluations
        # 3. Keep the best setting for each dimension
        # 4. Then refine with a finer grid around the found integers
        # 5. Finally, Nelder-Mead polish
        
        # Actually, let's be smarter:
        # Evaluate at a set of candidate points that cover the likely basins.
        
        # Candidate points: all combinations of integers in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
        # That's 9^5 = 59049 points - still too many.
        
        # Better: Do coordinate-wise optimization.
        # For each coordinate, find the best integer value, holding others fixed.
        
        # Initialize at 0
        x = np.zeros(n)
        fx = evaluate(x)
        
        # Coordinate-wise search for the best integer
        # Try each coordinate, each integer from -4 to 4
        for i in range(n):
            best_val = x[i]
            best_f_i = fx
            for val in range(-4, 5):
                if val == x[i]:
                    continue
                x_test = x.copy()
                x_test[i] = float(val)
                f_test = evaluate(x_test)
                if f_test < best_f_i:
                    best_f_i = f_test
                    best_val = val
            x[i] = best_val
            fx = best_f_i
            
            if calls >= budget:
                break
        
        # Now x is at some integer vector. Refine within the basin.
        # Use a finer grid around the current integer point.
        # For each dimension, try x_i + delta for delta in {-0.5, -0.25, 0, 0.25, 0.5}
        # But this is 5^5 = 3125 points - still a lot.
        
        # Instead, do coordinate-wise refinement:
        # For each dimension, try a few values around the current integer.
        
        for i in range(n):
            base = x[i]
            best_val = base
            best_f_i = fx
            for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                if abs(delta) < 1e-10:
                    continue
                val = base + delta
                if val < -5.0 or val > 5.0:
                    continue
                x_test = x.copy()
                x_test[i] = val
                f_test = evaluate(x_test)
                if f_test < best_f_i:
                    best_f_i = f_test
                    best_val = val
            x[i] = best_val
            fx = best_f_i
            
            if calls >= budget:
                break
        
        # Now do a second pass of coordinate-wise refinement with smaller steps
        for i in range(n):
            base = x[i]
            best_val = base
            best_f_i = fx
            for delta in [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]:
                val = base + delta
                if val < -5.0 or val > 5.0:
                    continue
                x_test = x.copy()
                x_test[i] = val
                f_test = evaluate(x_test)
                if f_test < best_f_i:
                    best_f_i = f_test
                    best_val = val
            x[i] = best_val
            fx = best_f_i
            
            if calls >= budget:
                break
        
        # Update best
        if fx < best_f:
            best_f = fx
            best_x = x.copy()
        
        # Phase 2: Nelder-Mead refinement
        if calls < budget - 10 and time.time() < deadline - 5:
            x_nm = best_x.copy() if best_x is not None else x.copy()
            f_nm = best_f
            
            # Build initial simplex
            simplex = [x_nm.copy()]
            fvals = [f_nm]
            
            step_nm = 0.1
            for i in range(n):
                pert = x_nm.copy()
                if x_nm[i] + step_nm > 5.0:
                    pert[i] -= step_nm
                elif x_nm[i] - step_nm < -5.0:
                    pert[i] += step_nm
                else:
                    pert[i] += step_nm
                pert = np.clip(pert, -5.0, 5.0)
                
                f_pert = evaluate(pert)
                simplex.append(pert)
                fvals.append(f_pert)
                
                if calls >= budget:
                    break
            
            if len(simplex) < n + 1:
                return best_x, best_f

            # NM Iterations
            max_nm_iters = 100
            for _ in range(max_nm_iters):
                if time.time() > deadline - 5 or calls >= budget:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if np.std(fvals) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5.0, 5.0)
                fr = evaluate(xr)
                
                if calls >= budget:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5.0, 5.0)
                    fe = evaluate(xe)
                    if calls < budget:
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    beta = 0.5
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + beta * (xr - centroid), -5.0, 5.0)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - beta * (xr - centroid), -5.0, 5.0)
                    
                    fc = evaluate(xc)
                    
                    if calls >= budget:
                        break
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                            if calls >= budget:
                                break
            
            # Update best from NM
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        return best_x, best_f
