"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start L-BFGS-B with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin to avoid timeout (300s limit, 280s budget)
        deadline = start_time + 270.0

        # Weierstrass f16 is known to be difficult.
        # Strategy: Use a small number of diverse starting points and run L-BFGS-B from each.
        # L-BFGS-B is efficient for smooth-ish functions and handles bounds well.
        # We will allocate a fixed number of function evaluations per start.
        # To ensure early progress (AOCC), we first do a very cheap random scan.
        
        # Phase 1: Random Scan (10 points)
        # This ensures we have a decent initial best_f quickly.
        n_scan = 10
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        for _ in range(n_scan):
            if budget <= 0 or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        # Phase 2: Multi-start L-BFGS-B
        # We have budget - n_scan calls left.
        # L-BFGS-B uses a line search, so it can take many calls.
        # We need to limit the number of function evaluations per run.
        # scipy.optimize.minimize with method='L-BFGS-B' doesn't have a direct 'max_fun' 
        # in all versions or it's tricky to control exactly. 
        # However, we can use a wrapper to count calls and raise an exception to stop.
        
        # Let's define a wrapper for func that counts calls and stops if limit reached.
        # We will run several instances.
        
        # How many starts?
        # If we have ~490 calls left, and each L-BFGS-B run might take 50-100 calls,
        # we can afford 5-10 starts.
        # Let's try 5 starts.
        # Each start gets max_calls = (budget - n_scan) // 5
        # But we need to be careful with time.
        
        # A better approach:
        # Run L-BFGS-B with a callback that checks time and budget.
        
        class BudgetExceeded(Exception):
            pass

        calls_made = 0
        max_calls_per_run = max(10, (budget - n_scan) // 5)
        
        # Starting points for L-BFGS-B:
        # 1. The best point from scan
        # 2. 4 random points near the best point (local perturbations)
        # Or just 5 random points in the box?
        # Given the rotation/shift, random points in the box are safer for global coverage,
        # but L-BFGS-B is local.
        # Let's use:
        # 1. best_x
        # 2-5. Random points in [-4, 4]^dim (where optimum is likely)
        
        start_points = [best_x]
        for _ in range(4):
            # Sample near the center or randomly?
            # Random in [-4, 4] is good.
            p = rng.uniform(-4.0, 4.0, dim)
            start_points.append(p)

        for start_p in start_points:
            if budget <= 0 or time.time() > deadline:
                break
            
            # Reset call counter for this run? No, global budget.
            # We need a wrapper that respects the global budget and time.
            
            def wrapped_func(x):
                nonlocal budget, calls_made
                if budget <= 0:
                    raise BudgetExceeded()
                if time.time() > deadline:
                    raise BudgetExceeded()
                
                # Clip to box (L-BFGS-B handles bounds, but just in case)
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                budget -= 1
                calls_made += 1
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                return fx

            # Use L-BFGS-B
            # maxiter is not the same as max_fun. 
            # In L-BFGS-B, each iteration does a line search which can use multiple function evaluations.
            # We can limit maxiter to a small number to control the number of evaluations roughly.
            # Or use the callback to stop.
            
            # Let's use a callback to check if we've used too many calls for this specific run?
            # Actually, the global budget check in wrapped_func is sufficient.
            # We just need to make sure we don't hang.
            # L-BFGS-B is generally fast.
            
            try:
                from scipy.optimize import minimize
                
                # bounds
                bounds = [(-5.0, 5.0)] * dim
                
                # maxiter: limit iterations. If each iteration takes ~5-10 calls, 
                # and we have ~100 calls per run, maxiter=10-20 is reasonable.
                res = minimize(wrapped_func, start_p, method='L-BFGS-B', bounds=bounds, maxiter=20)
                
                # Update best from result
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except BudgetExceeded:
                pass
            except Exception:
                # Catch any other errors to keep going
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
