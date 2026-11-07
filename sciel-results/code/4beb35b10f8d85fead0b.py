import numpy as np
import time
import scipy.optimize as so

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        call_count = 0

        def check_budget():
            nonlocal call_count
            return call_count < budget and time.time() < deadline

        # Phase 1: Global Exploration with diverse starting points
        # Use a combination of random points and a coarse grid to ensure coverage
        n_initial = 30
        initial_points = []
        
        # Add center and corners/boundaries to ensure we check edges
        initial_points.append(np.zeros(d))
        for _ in range(5):
            initial_points.append(rng.uniform(-5, 5, d))
        
        # Add points from a Halton-like sequence for better coverage
        # Simple low-discrepancy generation
        def halton_sequence(n, d):
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(1, d + 1):
                    f = 1.0
                    i_temp = i
                    while i_temp > 0:
                        f /= j
                        i_temp = (i_temp - 1) // j
                    point.append(f)
                points.append(point)
            return np.array(points)
            
        halton_pts = halton_sequence(20, d)
        # Map [0,1] to [-5,5]
        halton_pts = halton_pts * 10.0 - 5.0
        initial_points.extend(halton_pts)
        
        # Evaluate initial points
        for x in initial_points:
            if not check_budget():
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local Refinement using Nelder-Mead from top candidates
        # We need to track the best points found in Phase 1 to restart from them
        # Since we only kept the single best, let's re-evaluate a small set of diverse points 
        # or simply use the best point found and a few random perturbations of it.
        
        # Better approach: Keep a list of the top few points from Phase 1
        # Let's re-run Phase 1 logic to store top K points if we have budget, 
        # or just use the best point and its neighbors.
        
        # Let's refine the best point found so far
        if best_x is not None and check_budget():
            # Run Nelder-Mead from the global best
            nm_calls = [0]
            def nm_func(x_nm):
                nonlocal call_count, best_f, best_x
                nm_calls[0] += 1
                if not check_budget():
                    return best_f # Return current best if out of budget to stop
                fx = func(x_nm)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_nm.copy()
                return fx
            
            try:
                so.minimize(
                    nm_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'maxfev': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 3: Multi-start Nelder-Mead from diverse local basins
        # The Katsuura function has many local minima. We need to explore them.
        # Let's pick several diverse points and run short Nelder-Meads from them.
        
        # Generate 5 diverse points around the domain
        diverse_starts = []
        diverse_starts.append(best_x.copy())
        for _ in range(4):
            # Random point, but biased towards center if we haven't found good values
            # Or just fully random
            pt = rng.uniform(-5, 5, d)
            diverse_starts.append(pt)
            
        # Also add points near the best solution found
        for i in range(3):
            pt = best_x + rng.normal(0, 1.0, d)
            pt = np.clip(pt, -5, 5)
            diverse_starts.append(pt)

        for start_pt in diverse_starts:
            if not check_budget():
                break
            
            # Quick check: if this start point is bad, skip detailed optimization?
            # Let's just run a short NM from each.
            nm_calls = [0]
            def nm_func_local(x_nm):
                nonlocal call_count, best_f, best_x
                nm_calls[0] += 1
                if not check_budget():
                    return best_f
                fx = func(x_nm)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_nm.copy()
                return fx

            try:
                so.minimize(
                    nm_func_local, 
                    start_pt, 
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': 50, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 4: Final Polish
        if best_x is not None and check_budget():
            # Use remaining budget for a high-precision polish
            remaining = budget - call_count
            if remaining > 10:
                nm_calls = [0]
                def nm_func_polish(x_nm):
                    nonlocal call_count, best_f, best_x
                    nm_calls[0] += 1
                    if not check_budget():
                        return best_f
                    fx = func(x_nm)
                    call_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x_nm.copy()
                    return fx

                try:
                    so.minimize(
                        nm_func_polish, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'maxfev': remaining, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                    )
                except Exception:
                    pass

        return best_x, best_f
