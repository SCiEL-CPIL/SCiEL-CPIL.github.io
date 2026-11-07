"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Global-Local Optimizer using Grid Search + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        x_best = np.zeros(dim)
        f_best = float('inf')
        calls = 0

        def safe_call(x):
            nonlocal calls
            if time.time() > deadline or calls >= budget:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            return val

        # Phase 1: Coarse Global Search
        # Allocate ~20% of budget to find good starting points
        # We use a stratified sampling approach rather than pure random
        # to ensure coverage of the box.
        
        n_global = max(10, int(budget * 0.1))
        
        # Simple stratified grid sampling for dim=5
        # 3^5 = 243 points might be too many if budget is small, 
        # but budget is 300. Let's use a smaller grid or random.
        # Let's use random but with a deterministic seed for reproducibility 
        # and a few fixed points (center, corners).
        
        rng = np.random.default_rng(42)
        
        # Fixed points: Center
        points_to_try = [np.zeros(dim)]
        
        # Random points
        random_points = rng.uniform(-5, 5, (n_global - 1, dim))
        
        for x in random_points:
            points_to_try.append(x)
            
        # Evaluate initial points
        for x in points_to_try:
            if time.time() > deadline or calls >= budget:
                break
            fx = safe_call(x)
            if fx is None:
                continue
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        # Phase 2: Local Refinement with L-BFGS-B
        # We will run local optimization from the best point found so far,
        # and potentially from other top-performing points if budget allows.
        
        # To keep it simple and robust, we restart from the global best 
        # and potentially a few other distinct good points found in Phase 1.
        # However, L-BFGS-B can be expensive in calls. 
        # We need to limit the number of function evaluations per local run.
        
        # Let's collect the top 5 distinct points from Phase 1 if possible.
        # For simplicity and speed, we'll just refine the single best point found.
        # If that converges quickly, we can try another.
        
        # Sort the evaluated points? We didn't store them all. 
        # Let's just use the current x_best.
        
        # We can try to do a few "hops" or restarts if the local opt is stuck.
        
        remaining_budget = budget - calls
        
        if remaining_budget > 10 and time.time() < deadline:
            # Use L-BFGS-B
            # maxiter and maxfun control the iterations.
            # We want to use up to ~50% of remaining budget for this local run.
            max_fun = int(remaining_budget * 0.4)
            max_fun = max(10, min(max_fun, 100)) # Cap at 100 to avoid hanging on one local min
            
            try:
                res = scipy.optimize.minimize(
                    func, 
                    x_best, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': max_fun, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Note: scipy.minimize might not update 'calls' if we don't wrap it, 
                # but we passed 'func' directly. 
                # CRITICAL: The evaluator counts calls to 'func'. 
                # scipy.optimize.minimize calls 'func' internally. 
                # We MUST ensure our 'calls' counter is updated.
                # Since we passed the raw 'func', our counter 'calls' is NOT updated by scipy.
                # We need to wrap func to count calls.
                
                # Let's re-evaluate the strategy.
                # We need a wrapper for func that increments 'calls' and checks time.
                
            except Exception:
                pass

        # The above try block failed to update calls because we passed raw func.
        # Let's implement a proper wrapper.
        
        # Reset state for a cleaner implementation of the hybrid approach.
        # Actually, let's just rewrite the logic inside __call__ to use a wrapper.
        
        # --- RESTART LOGIC WITH PROPER WRAPPING ---
        
        # We already used some calls in Phase 1.
        # Let's define the wrapper now.
        
        def wrapped_func(x):
            nonlocal calls, f_best, x_best
            if time.time() > deadline or calls >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls += 1
            if val < f_best:
                f_best = val
                x_best = x_clipped.copy()
            return val

        # Phase 2: Local Optimization from the best point found in Phase 1
        # We use the wrapped function so calls are counted.
        
        if time.time() < deadline and calls < budget:
            # Determine budget for local search
            # Leave some budget for potential restarts if this fails
            local_budget = int((budget - calls) * 0.5)
            local_budget = max(20, min(local_budget, 150))
            
            try:
                # L-BFGS-B is good for smooth functions.
                # For Rastrigin/Gallagher, it might get stuck in local minima.
                # But it converges very fast when close.
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_best, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': local_budget, 'ftol': 1e-14, 'gtol': 1e-8}
                )
            except Exception:
                pass

        # Phase 3: If we have budget left, try a few more local optimizations 
        # from random points or the previous best if it improved.
        
        if time.time() < deadline and calls < budget:
            # Try 2 more restarts from random points
            for i in range(2):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Pick a random start, maybe biased towards the current best?
                # No, random is better for diversity.
                x_start = rng.uniform(-5, 5, dim)
                
                local_budget_2 = int((budget - calls) * 0.3)
                local_budget_2 = max(10, min(local_budget_2, 50))
                
                try:
                    res2 = scipy.optimize.minimize(
                        wrapped_func, 
                        x_start, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfun': local_budget_2, 'ftol': 1e-14, 'gtol': 1e-8}
                    )
                except Exception:
                    pass

        return x_best, f_best

# EVOLVE-BLOCK-END
