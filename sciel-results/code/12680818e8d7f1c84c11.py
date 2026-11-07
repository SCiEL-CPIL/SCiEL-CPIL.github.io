"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard: 330s limit, leave 10s margin
        start_time = time.time()
        deadline = start_time + 320.0

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # Helper to evaluate and track budget
        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation at origin (often a good heuristic for BBOB if shifts are moderate)
        # But since we don't know the shift, we will do a few random starts + origin
        initial_points = [np.zeros(dim)]
        
        # Add a few random points to cover the space
        rng = np.random.default_rng(42)
        for _ in range(4):
            initial_points.append(rng.uniform(-4, 4, dim))

        # Evaluate all initial points
        for x_init in initial_points:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            eval_point(x_init)

        # If we haven't used up the budget, perform Nelder-Mead from the best initial point
        # and potentially restart if we get stuck or time allows.
        
        # Nelder-Mead parameters
        # For dim=5, standard NM works well.
        # maxiter is a safety cap for scipy, but we control the budget via the function calls.
        # However, scipy's NM doesn't have a direct 'budget' callback that stops it mid-iteration 
        # easily without wrapping. We will wrap the function to check time/budget.
        
        def wrapped_func(x):
            nonlocal best_x, best_f, calls_used
            if time.time() > deadline:
                # Raise an exception to break out of scipy if we hit time limit
                # But we want to keep the best found so far.
                # It's safer to just return a very large value and let it finish, 
                # or check time in the loop.
                # Actually, scipy.optimize.minimize doesn't handle mid-run budget checks well.
                # We will implement a simple manual Nelder-Mead or use scipy with a tight maxiter 
                # and restart.
                pass
            
            if calls_used >= budget:
                return 1e100 # Large value to discourage further exploration
            
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: 
        # 1. Start NM from the best initial point found.
        # 2. If time/budget remains, start NM from other initial points if they are competitive.
        # 3. Since NM is fast, we can run multiple instances.
        
        # Sort initial points by fitness to prioritize good starts
        # We already evaluated them. Let's re-evaluate to get their current fitness for sorting?
        # No, we can just pick the best one first.
        
        # Find the best initial point
        init_fits = []
        for x_init in initial_points:
            # We already evaluated these, but we didn't store the specific fitness for sorting easily 
            # without re-evaluating or storing it. Let's just pick the best_x so far.
            pass
        
        # Let's run NM from the current best_x
        if calls_used < budget and time.time() < deadline:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except Exception:
                pass

        # If we still have budget and time, try another restart from a different point
        # to escape potential local minima.
        if calls_used < budget and time.time() < deadline:
            # Pick a random point different from best_x
            x_restart = rng.uniform(-4, 4, dim)
            # Ensure it's not too close to best_x
            if np.linalg.norm(x_restart - best_x) > 1.0:
                try:
                    res2 = scipy.optimize.minimize(
                        wrapped_func, 
                        x_restart, 
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                except Exception:
                    pass
        
        # If we still have budget and time, one more restart
        if calls_used < budget and time.time() < deadline:
            x_restart = rng.uniform(-4, 4, dim)
            if np.linalg.norm(x_restart - best_x) > 1.0:
                try:
                    res3 = scipy.optimize.minimize(
                        wrapped_func, 
                        x_restart, 
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                    )
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
