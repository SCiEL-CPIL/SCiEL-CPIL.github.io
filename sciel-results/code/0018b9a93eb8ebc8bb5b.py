import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = 300.0  # Hard limit is 330s, use 300s for safety

    def __call__(self, func):
        self.start_time = time.time()
        
        # Internal counter for the black-box function
        def wrapped_func(x):
            # Check time limit
            if time.time() - self.start_time > self.deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Check budget
            if self.calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            
            self.calls_made += 1
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            
            fx = func(x)
            
            # Update best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            return fx

        # Multi-start L-BFGS-B strategy
        # We will try multiple random starting points.
        # Each local optimization run is limited in iterations/calls.
        
        # Heuristic: Use ~30-40 calls per restart on average.
        # With 500 calls, we can do ~10-15 restarts.
        
        rng = np.random.default_rng(42)
        
        # Initial random points
        # Let's do a few quick random evaluations first to find a decent start?
        # Or just start L-BFGS-B directly from random points.
        
        # Strategy:
        # 1. Pick a random start.
        # 2. Run L-BFGS-B with maxiter such that it doesn't consume all budget.
        # 3. If it converges or uses up its allocated sub-budget, pick a new random start.
        
        # To ensure early progress, we should start local optimization immediately.
        
        num_restarts = 0
        max_restarts = 20 # Upper bound, will stop earlier if budget/time runs out
        
        while num_restarts < max_restarts:
            if self.calls_made >= self.budget:
                break
            
            if time.time() - self.start_time > self.deadline - 5: # 5s margin
                break
            
            # Choose a starting point
            # Uniform in [-5, 5]
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Run L-BFGS-B
            # We need to limit the number of calls in this specific run
            # so we can do multiple restarts.
            # Let's allocate a dynamic sub-budget.
            # If we have 500 calls left, and 10 restarts left, ~50 calls each.
            remaining_calls = self.budget - self.calls_made
            restarts_left = max_restarts - num_restarts
            if restarts_left <= 0:
                break
            
            # Reserve some calls for future restarts?
            # Simple heuristic: allow up to 40 calls per restart, or all remaining if last.
            # But L-BFGS-B might converge in 20.
            sub_budget = min(40, remaining_calls)
            
            try:
                # L-BFGS-B options
                # maxiter is iteration count, not function eval count.
                # In 5D, 1 iteration ~ 6-10 function evals?
                # Let's use maxiter=10 which is roughly 60-100 evals? Too high.
                # Better to use a callback to count function evaluations.
                
                eval_count_in_run = 0
                
                def callback(xk):
                    nonlocal eval_count_in_run
                    # This callback is called after each iteration, not each function eval.
                    # It's hard to strictly limit function evals with scipy's minimize
                    # without a custom wrapper that counts.
                    # Our wrapped_func counts them.
                    # We can check self.calls_made here.
                    if self.calls_made >= self.budget - 5: # Leave 5 for safety
                        return True # Stop optimization
                    if time.time() - self.start_time > self.deadline - 5:
                        return True
                    return False

                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback}
                )
                
                # Update best from result if valid
                if res.success or res.fun < self.best_f:
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x.copy()
                        
            except (TimeoutError, ValueError):
                break
            except Exception:
                # Catch any other errors from optimizer
                break
            
            num_restarts += 1
            
            # If the optimization used very few calls (e.g. converged quickly), 
            # we might want to restart immediately.
            # If it used all sub_budget, we restart.
            
        return self.best_x, self.best_f
