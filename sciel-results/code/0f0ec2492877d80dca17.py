from scipy.optimize import minimize
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget or time.time() > deadline:
                return best_f
            calls_made += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Global Search with Nelder-Mead from multiple starting points
        # Rastrigin has many local minima. We need good coverage.
        
        # Generate diverse starting points
        # 1. Origin
        # 2. Random uniform points
        # 3. Points aligned with axes (to help escape some local minima structures)
        
        rng = np.random.default_rng(123)
        
        start_points = [np.zeros(n)]
        
        # Add random points
        num_random_starts = 12
        for _ in range(num_random_starts):
            start_points.append(rng.uniform(-5, 5, n))
            
        # Add axis-aligned points (e.g., +/-1, +/-2, +/-4 on each axis)
        # This helps if the shift/rotation isn't too severe
        for axis in range(n):
            for val in [-4, -2, -1, 1, 2, 4]:
                p = np.zeros(n)
                p[axis] = val
                start_points.append(p)
        
        # Shuffle starts to randomize which ones get budget if we run out
        # But keep origin first? No, random is fine. Actually, origin is a decent bet for Rastrigin (global min is at 0 before shift).
        # Let's sort by a heuristic? No, just shuffle.
        indices = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in indices]
        
        # We have ~1 + 12 + 30 = 43 start points.
        # Budget is 500.
        # If we use ~10 evals per start, that's 430. Leaves 70 for polish.
        # NM converges fast. Let's give each start a small budget, say 10-15 evals.
        
        evals_per_start = 10
        max_starts_to_try = total_budget // evals_per_start
        
        for i, x0 in enumerate(start_points):
            if i >= max_starts_to_try:
                break
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            
            # Run Nelder-Mead from x0
            # We use a wrapper that respects the global budget
            nm_calls = [0]
            def nm_func(x):
                nonlocal nm_calls
                # Check internal limit
                if nm_calls[0] >= evals_per_start:
                    return best_f
                # Global check is handled in evaluate, but we need to stop NM early
                # NM doesn't have a direct "stop if budget" in scipy, so we return a value that triggers convergence?
                # Actually, if we return best_f (which is likely smaller than the current NM point if we found something good elsewhere),
                # it might not stop it. 
                # Better: rely on maxiter.
                nm_calls[0] += 1
                return evaluate(x)

            try:
                # Nelder-Mead options
                # maxiter is the number of function evaluations? No, it's the number of iterations.
                # For NM, one iteration is roughly N+1 function evaluations.
                # So to limit function evaluations to ~evals_per_start, set maxiter = evals_per_start // (n+1)
                # But we want to use the budget.
                # Let's just set maxiter high enough and let the budget wrapper cut it off.
                # But NM doesn't know about the global budget inside the loop unless the function raises or returns specific values.
                # The `evaluate` function returns best_f if budget exceeded.
                # If NM keeps going, it wastes CPU time but no function calls.
                # However, if time runs out, we break the outer loop.
                
                # To be safe, set maxiter to a reasonable number.
                max_iter_nm = 50 # This is iterations, not evals. 50 * 6 = 300 evals worst case if it doesn't stop.
                # But our wrapper cuts it at evals_per_start *global* or best_f.
                # Actually, if nm_calls[0] >= evals_per_start, we return best_f.
                # If best_f is very good, NM might converge. If not, it might loop.
                # Let's just trust the evals_per_start limit in the wrapper.
                
                res = minimize(nm_func, x0, method='Nelder-Mead',
                               options={'maxiter': max_iter_nm, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
                
            except Exception:
                pass

        # Phase 2: Local Refinement (Polish) on the best point found
        # If we have budget left, do a high-precision polish.
        if calls_made < total_budget and time.time() < deadline and best_x is not None:
            remaining_budget = total_budget - calls_made
            if remaining_budget > 20:
                # Use Powell or NM with tighter tolerances?
                # NM is good for smooth functions. Rastrigin is smooth but multimodal.
                # Since we are at a local min, NM should work well.
                
                polish_calls = [0]
                def polish_func(x):
                    nonlocal polish_calls
                    if polish_calls[0] >= remaining_budget:
                        return best_f
                    polish_calls[0] += 1
                    return evaluate(x)

                try:
                    # More iterations, tighter tolerances
                    res = minimize(polish_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': remaining_budget // (n + 1) + 10, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10})
                except Exception:
                    pass

        # Phase 3: If still budget left, try a few more random restarts or just random sampling?
        # Given Rastrigin, more random restarts might find a better basin.
        # But NM from good basins is fast.
        # Let's just do a final random check to see if we missed something obvious.
        
        while calls_made < total_budget and time.time() < deadline:
            x = rng.uniform(-5, 5, n)
            f_val = evaluate(x)
            if f_val >= best_f: # No improvement
                # Try a small perturbation around best_x?
                x_pert = best_x + rng.normal(0, 0.1, n)
                evaluate(x_pert)
            else:
                # Found something better! Run NM from there.
                nm_calls = [0]
                def nm_func2(x):
                    nonlocal nm_calls
                    if nm_calls[0] >= 20:
                        return best_f
                    nm_calls[0] += 1
                    return evaluate(x)
                try:
                    minimize(nm_func2, x, method='Nelder-Mead',
                             options={'maxiter': 20, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
                except Exception:
                    pass

        return best_x, best_f
