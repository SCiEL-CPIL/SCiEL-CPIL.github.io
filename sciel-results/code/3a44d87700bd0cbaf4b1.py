"""
Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start Powell's method
with dense, stratified Latin Hypercube initialization.
"""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Dense Latin Hypercube Sampling for global basin identification
        # Use stratified LHS to ensure coverage of the entire [-5, 5]^dim box
        n_lhs = 50  # Number of LHS points for initial exploration
        
        def generate_lhs(n, dim):
            """Generate n points in [0, 1]^dim using Latin Hypercube Sampling."""
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = np.random.permutation(n)
                points[:, d] = (perm + np.random.uniform(0, 1, n)) / n
            return points

        lhs_points = generate_lhs(n_lhs, dim)
        # Map from [0, 1] to [-5, 5]
        lhs_points = lhs_points * 10 - 5

        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_points[i]
            f = func(x)
            calls += 1
            update_best(x, f)

        # Phase 2: Multi-start Powell's method
        # Select top candidates from LHS and run Powell from each
        # Powell's method is efficient for smooth, separable-like functions
        # Lunacek bi-Rastrigin has smooth basins with periodic ridges
        
        # Get top 5 points from LHS as starting points for Powell
        # We need to track which points were evaluated
        evaluated_points = []
        for i in range(n_lhs):
            if calls > i:  # This is a simplification; in reality we track all
                pass
        
        # Re-evaluate to get sorted list (we already have them, but let's be explicit)
        # Actually, we can just use the best few from our tracked evaluations
        # Let's collect all evaluated points and their values
        all_points = []
        all_values = []
        
        # We need to re-collect since we didn't store them in the loop above
        # Let's redo the LHS evaluation properly storing results
        # Wait, we already evaluated them. Let's just use the best ones we know.
        # Actually, let's restart the LHS part to store properly for sorting.
        
        # Let's clear and redo with proper storage
        calls = 0
        best_x = None
        best_f = np.inf
        
        lhs_results = []
        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_points[i]
            f = func(x)
            calls += 1
            lhs_results.append((x, f))
            update_best(x, f)
        
        # Sort by function value
        lhs_results.sort(key=lambda t: t[1])
        
        # Select top 5 starting points for Powell
        n_powell_starts = min(5, len(lhs_results))
        powell_starts = [lhs_results[i][0] for i in range(n_powell_starts)]
        
        # Run Powell's method from each starting point
        # Powell uses line searches along directions, very efficient for smooth functions
        max_evals_per_powell = 80  # Limit evaluations per Powell run
        
        for start_x in powell_starts:
            if calls >= budget or time.time() > deadline:
                break
            
            # Use scipy's Powell method with bounds
            # Powell doesn't support bounds directly, but we can use a wrapper
            # or rely on the fact that Powell stays near the minimum if it's in bounds
            
            # Actually, let's use a bounded version or just clip
            # SciPy's Powell doesn't have bounds parameter in all versions
            # Let's use Nelder-Mead instead which is more robust, or implement bounded Powell
            # Actually, let's try Powell and hope it stays in bounds, or use a penalty
            
            # Better: use Nelder-Mead with adaptive settings, which is robust
            # Or use Powell with a wrapper that clips
            
            # Let's use Powell with a wrapper function that clips to bounds
            def bounded_func(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                update_best(x_clipped, f)
                return f
            
            try:
                result = minimize(bounded_func, start_x, method='Powell',
                                 options={'maxiter': 50, 'maxfev': max_evals_per_powell,
                                          'xtol': 1e-8, 'ftol': 1e-8})
            except Exception:
                pass
            
            # After Powell, do a local refinement with Nelder-Mead for precision
            if calls < budget and time.time() < deadline:
                current_best = best_x if best_x is not None else start_x
                try:
                    result2 = minimize(bounded_func, current_best, method='Nelder-Mead',
                                      options={'maxiter': 100, 'maxfev': 30,
                                               'xatol': 1e-10, 'fatol': 1e-10,
                                               'adaptive': True})
                except Exception:
                    pass
        
        # Phase 3: Final local refinement with Nelder-Mead from the global best
        if calls < budget and time.time() < deadline and best_x is not None:
            def final_func(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls += 1
                update_best(x_clipped, f)
                return f
            
            try:
                minimize(final_func, best_x, method='Nelder-Mead',
                        options={'maxiter': 200, 'maxfev': 50,
                                 'xatol': 1e-12, 'fatol': 1e-12,
                                 'adaptive': True})
            except Exception:
                pass
        
        # Phase 4: If we have remaining budget, do additional random restarts with local optimization
        remaining = budget - calls
        if remaining > 20 and time.time() < deadline:
            # Do a few more random starts with Nelder-Mead
            n_extra = min(3, remaining // 30)
            for _ in range(n_extra):
                if calls >= budget or time.time() > deadline:
                    break
                
                x_start = np.random.uniform(-5, 5, size=dim)
                def extra_func(x):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    x_clipped = np.clip(x, -5, 5)
                    f = func(x_clipped)
                    calls += 1
                    update_best(x_clipped, f)
                    return f
                
                try:
                    minimize(extra_func, x_start, method='Nelder-Mead',
                            options={'maxiter': 50, 'maxfev': 20,
                                     'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True})
                except Exception:
                    pass
        
        return best_x, best_f
