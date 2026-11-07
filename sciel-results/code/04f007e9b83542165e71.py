"""Rewritten optimizer for BBOB-style Rastrigin rotated (f15)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start Nelder-Mead with coordinate descent refinement for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def _check_budget():
            nonlocal calls
            return calls < budget and time.time() < deadline

        def _call(func, x):
            nonlocal calls
            if not _check_budget():
                return None
            val = float(func(x))
            calls += 1
            return val

        # Phase 1: Multi-start Nelder-Mead from diverse initial points
        # For Rastrigin, minima are at integer lattice points (shifted/rotated)
        # Sample integer points and random points to find good basins
        
        n_starts = 8
        start_points = []
        
        # Integer points in [-4, 4]^5 - sample a subset
        int_coords = list(range(-4, 5))
        # Sample random integer combinations
        for _ in range(4):
            pt = np.array([rng.choice(int_coords) for _ in range(dim)], dtype=float)
            start_points.append(pt)
        
        # Random continuous points
        for _ in range(4):
            pt = rng.uniform(-4.5, 4.5, size=dim)
            start_points.append(pt)
        
        # Evaluate all start points and keep the best ones
        start_results = []
        for pt in start_points:
            if not _check_budget():
                break
            val = _call(func, pt)
            if val is not None:
                start_results.append((val, pt.copy()))
        
        if not start_results:
            # Fallback: single random point
            x = rng.uniform(-5, 5, size=dim)
            fx = _call(func, x)
            if fx is not None:
                best_x, best_f = x.copy(), fx
            return best_x if best_x is not None else np.zeros(dim), best_f
        
        # Sort by value, take top 4 for Nelder-Mead
        start_results.sort(key=lambda t: t[0])
        top_starts = start_results[:min(4, len(start_results))]
        
        # Phase 2: Nelder-Mead from each top start
        # Distribute remaining budget among starts
        remaining = budget - calls
        budget_per_start = max(20, remaining // len(top_starts))
        
        for val, start_pt in top_starts:
            if not _check_budget():
                break
            
            # Nelder-Mead with scipy
            x0 = start_pt.copy()
            f0 = val
            best_local_x = x0
            best_local_f = f0
            
            # Use scipy's minimize with Nelder-Mead
            # We need to wrap func to track calls and check budget
            def _wrapped_func(x):
                nonlocal calls, best_local_x, best_local_f
                if not _check_budget():
                    return best_local_f  # Return current best to stop
                val = float(func(x))
                calls += 1
                if val < best_local_f:
                    best_local_f = val
                    best_local_x = x.copy()
                return val
            
            try:
                result = scipy.optimize.minimize(
                    _wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_start,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_local_f:
                    best_local_f = result.fun
                    best_local_x = result.x.copy()
            except Exception:
                pass
            
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()
        
        # Phase 3: Coordinate descent refinement from the best point
        if best_x is not None and _check_budget():
            x = best_x.copy()
            fx = best_f
            step = 0.1
            max_cd_iters = min(100, budget - calls)
            
            for _ in range(max_cd_iters):
                if not _check_budget():
                    break
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        y = x.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = _call(func, y)
                        if fy is None:
                            break
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    if not _check_budget():
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
                else:
                    step *= 1.2
                    step = min(step, 0.5)
        
        return best_x if best_x is not None else np.zeros(dim), best_f
# EVOLVE-BLOCK-END
