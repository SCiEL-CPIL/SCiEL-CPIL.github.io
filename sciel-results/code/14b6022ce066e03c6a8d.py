"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Multi-Start Nelder-Mead optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_f = np.inf
        best_x = np.zeros(n)
        calls_made = 0
        
        # We will use a fixed seed for the random restarts to ensure deterministic behavior
        # which is often preferred in these black-box benchmarks unless specified otherwise.
        # However, using a random seed based on time might be better for exploration diversity
        # across different instances if the evaluator runs them in parallel or separately.
        # The prompt says "hidden seeded shift", implying the function is fixed per instance.
        # Using np.random.default_rng() without a seed will use system entropy.
        rng = np.random.default_rng()

        # Strategy: Multi-start Nelder-Mead
        # Divide budget into restarts.
        # Each restart gets a chunk of the budget.
        # If a restart finishes early (convergence), we start a new one.
        # If a restart uses all its budget, we start a new one.
        
        # How many restarts? 
        # With dim=5, Nelder-Mead is fast. 
        # Let's allocate a dynamic budget per restart.
        # Start with a generous budget for the first few, then smaller ones?
        # Or just fixed chunks?
        # Fixed chunks of 20 calls: 500 / 20 = 25 restarts.
        # This allows 25 distinct local minima to be explored.
        
        calls_per_restart = 20
        
        # Initial evaluation at center (0,0,0,0,0) is a good idea for f20?
        # f20 (Schwefel) has a global minimum at a specific point, but many local minima.
        # (0,0,0,0,0) is not the global min.
        # We will let the random restarts handle it.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Determine budget for this restart
            remaining = budget - calls_made
            current_budget = min(calls_per_restart, remaining)
            
            # Generate a random starting point in [-5, 5]^n
            # Using a slightly smaller box for starts might be better? 
            # No, the global min is in [-4, 4]^5, but starts should cover the space.
            x0 = rng.uniform(-5, 5, size=n)
            
            # Wrapper to count calls and handle exceptions/time
            def wrapped_func(x):
                nonlocal calls_made
                if time.time() > deadline:
                    raise Exception("Time limit exceeded")
                if calls_made >= budget:
                    raise Exception("Budget exceeded")
                
                # Clip input to be safe, though scipy might handle it?
                # The task says func clips. We trust func.
                try:
                    f_val = func(x)
                    calls_made += 1
                    return f_val
                except Exception:
                    # If func throws, return a large penalty
                    calls_made += 1
                    return 1e10

            # Run Nelder-Mead
            # options: maxiter is not the same as function calls.
            # maxiter is the number of iterations. Each iteration can take multiple calls.
            # For dim=5, one NM iteration takes approx 2*n + 1 = 11 calls?
            # Actually, it varies.
            # It is safer to use a custom loop or rely on maxiter being small.
            # Let's estimate: 20 calls / 5 dim ~ 4 iterations?
            # Let's set maxiter to a value that roughly fits current_budget.
            # A rough estimate is that NM takes ~n calls per iteration on average.
            # So maxiter = current_budget // n is a conservative estimate.
            max_iter = max(1, current_budget // n)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # Check result
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                # If an exception occurs (e.g. timeout), we keep the best found so far
                # and break out of the restart loop if it's a timeout
                if time.time() > deadline:
                    break
                # Otherwise, continue to next restart if budget allows

        return best_x, best_f
# EVOLVE-BLOCK-END
