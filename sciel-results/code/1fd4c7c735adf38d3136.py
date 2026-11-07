"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Hybrid optimizer using Random Restarts and L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0
        start_time = time.time()
        
        # Safety margin for time (300s limit, keep 30s buffer)
        time_limit = 270.0

        # We will use a strategy of multiple restarts.
        # L-BFGS-B is very efficient. 
        # Let's try to use the budget wisely.
        # If we use 100 calls per restart, we can do 5 restarts.
        # If we use 50 calls per restart, we can do 10 restarts.
        # Given the step ellipsoidal nature, we need to find the basin.
        # Random restarts help.
        
        # Let's allocate a specific number of calls per local search.
        # If the first few don't work, we might need more restarts.
        # Let's try 5 restarts with 100 calls each, or adaptively.
        
        # A better approach: 
        # 1. Try a few random points with L-BFGS-B.
        # 2. If we find a very good value (close to expected min), stop.
        # 3. Otherwise, continue restarting.
        
        # Since we don't know f_opt, we just minimize.
        # Let's set max_evals per local run to allow multiple restarts.
        # 500 total calls.
        # Let's do 10 restarts of 50 calls each? 
        # Or 5 restarts of 100 calls each?
        # L-BFGS-B often converges in < 50 iterations for smooth 5D functions.
        # Let's try 10 restarts, max 40 calls each, leaving 100 for extra if needed?
        # Actually, let's just loop until budget is exhausted.
        
        while calls_made < self.budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate a random starting point
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Define a wrapper to count calls
            # Note: scipy minimize uses a callback or just counts internally?
            # It's safer to wrap the function.
            
            local_calls = [0]
            
            def wrapped_func(x):
                nonlocal calls_made
                nonlocal best_x
                nonlocal best_f
                nonlocal local_calls
                
                # Check budget and time
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    # Return a value that discourages further exploration?
                    # Or just raise an exception? 
                    # Better to just return the current best or a large value.
                    # Actually, if we exceed budget, we should stop.
                    # But scipy might call it multiple times in one step?
                    # Let's just return a large value to stop improvement.
                    return 1e10
                
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                local_calls[0] += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                return fx

            # Determine max evaluations for this local run
            # Remaining budget
            remaining = self.budget - calls_made
            if remaining <= 0:
                break
            
            # Cap the local run to allow restarts
            # If we are early, we can afford more calls per run to converge.
            # If we are late, we do shorter runs.
            # Let's try a fixed number of calls per restart, e.g., 50.
            # 500 / 50 = 10 restarts.
            max_local_evals = 50
            
            # If remaining budget is less than max_local_evals, use remaining
            if remaining < max_local_evals:
                max_local_evals = remaining
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_local_evals, 'maxfun': max_local_evals}
                )
                
                # The minimize function might not use all maxfun if it converges earlier
                # Update best if the result is better (though wrapped_func already updates)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If an exception occurs, we might have used some calls.
                # Continue to next restart if budget allows.
                pass
            
            # If we converged very well, we might break early?
            # Hard to know without f_opt. 
            # But if best_f is very small, maybe? 
            # Let's stick to using the full budget unless time runs out.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
