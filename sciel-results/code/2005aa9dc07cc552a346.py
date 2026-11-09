"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start CMA-ES optimizer with global budget control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s safety margin.
        deadline = start_time + 280 
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        # Use a deterministic seed for reproducibility of the search strategy
        rng = np.random.default_rng(123)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            # Hard budget check
            if call_count >= self.budget:
                return np.inf
            # Hard time check
            if time.time() > deadline:
                return np.inf
            
            call_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f_val = func(x_clipped)
            except Exception:
                return np.inf
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Strategy: Multi-start CMA-ES
        # BBOB f4 is multi-modal with a shift/rotation.
        # CMA-ES is robust to rotations and shifts.
        # We will allocate the budget to several independent CMA-ES runs.
        
        # dim=5.
        # Population size for CMA-ES: default is lambda = int(4 + 3 * log(n)) for n>1?
        # Actually scipy CMA-ES default lambda is int(4 + 3 * log(n)).
        # For n=5, log(5)=1.6, 4+4.8=8.8 -> 8? Or maybe (4+ln(5))*5 = 28?
        # Let's check scipy docs mentally: default is often around 4-10 for small n, or larger.
        # To be safe and allow convergence, we want enough evaluations per run.
        
        # Total budget 500.
        # If we do 5 restarts, we have 100 evals each.
        # CMA-ES usually needs ~10-20 * lambda * n to converge well? 
        # 100 evals is tight but might find a good local minimum if initialized well.
        
        # Better: 3 restarts of ~166 evals each.
        # Or 2 restarts of 250 evals each.
        
        # Let's try 4 restarts.
        num_restarts = 4
        evals_per_restart = self.budget // num_restarts # 125
        
        # Generate starting points.
        # Instead of pure random, we can use a low-discrepancy sequence or just random.
        # Random is fine with a fixed seed.
        # To ensure diversity, we can use different seeds for each restart or just sequential rng.
        
        for i in range(num_restarts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Random start point
            x0 = rng.uniform(-5, 5, self.dim)
            
            # CMA-ES settings
            # maxiter is the number of iterations (generations).
            # Each iteration consumes ~lambda evaluations.
            # We don't know exact lambda scipy uses, but typically ~10-30 for dim=5.
            # If lambda=10, 125 evals = 12 iterations.
            # If lambda=20, 125 evals = 6 iterations.
            # This might be too few for CMA-ES to converge from a random start to the global min.
            
            # Alternative: Use a smaller number of restarts but larger budget each.
            # Let's switch to 2 restarts of 250 evals.
            
            # Wait, the loop above is 4 restarts. Let's adjust logic to 2 restarts for better convergence per run.
            
            pass # Placeholder, logic below

        # Re-implement with 2 restarts to allow deeper search.
        num_restarts = 2
        # We already used 0 calls.
        
        # Start point 1: Origin (0,0,0,0,0) is often a good heuristic for BBOB if shift is not huge, 
        # but shift is in [-4,4]. Random is safer.
        
        for i in range(num_restarts):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Diversify starting points
            # Run 1: Random
            # Run 2: Random
            x0 = rng.uniform(-5, 5, self.dim)
            
            # We want to use ~250 evals.
            # Set maxiter high enough that it doesn't stop prematurely due to iteration limit,
            # but rely on the wrapper's budget cap to stop it.
            # However, if it stops early due to convergence, that's good.
            
            # To prevent it from running forever if it doesn't converge, 
            # and to ensure we use the budget, we set maxiter to a large number.
            # But if it doesn't converge, it will hit the call_count limit in wrapped_func.
            
            # Potential issue: CMA-ES might check for convergence and stop before using all budget.
            # This is fine.
            
            # Another issue: CMA-ES might throw an exception if it gets stuck? 
            # wrapped_func returns inf, which might cause issues? 
            # Usually optimizers handle inf gracefully by stopping or ignoring.
            
            # Let's add a callback to stop if time is up? 
            # SciPy CMA-ES doesn't support callback return False to stop in older versions?
            # Actually, `callback` in minimize can return False to stop.
            
            def callback(xk):
                return time.time() < deadline and call_count < self.budget

            try:
                # options: 
                # 'maxiter': 1000 (high, let budget/time limit it)
                # 'xatol', 'fatol': default
                res = minimize(wrapped_func, x0, method='CMA', 
                               options={'maxiter': 1000, 'verbose': 0},
                               callback=callback)
                
                # res.fun might be inf if we hit budget limit mid-run?
                # But we track best_f in wrapper.
                
            except Exception as e:
                # If CMA-ES fails, continue to next restart
                pass
            
            # Check if we should do a final polish?
            # If we have budget left, maybe run a local optimizer on the best point?
            # CMA-ES is already a local optimizer.
        
        # If we have significant budget left (e.g. < 400 calls used), 
        # we could do more restarts. 
        # But 2 restarts of CMA-ES should use most of the budget if they don't converge fast.
        
        # If we converged fast (e.g. used 100 calls in first run), 
        # we have 400 left. 
        # We should use them!
        
        # Let's add a loop for more restarts if budget remains.
        
        # Reset loop logic to be dynamic.
        
        # Actually, the previous loop only ran 2 times.
        # Let's change the structure to a while loop.
        
        # Rewriting the main loop to be dynamic:
        
        # We already did 2 runs in the loop above? No, I put `pass` and then re-implemented.
        # The code above with `num_restarts=2` and `for i in range(num_restarts)` is the active one.
        # It runs exactly 2 times.
        
        # If budget remains, let's do more random restarts.
        
        while call_count < self.budget and time.time() < deadline:
            # Check if we have enough budget for a meaningful CMA-ES run (e.g. 50 evals)
            if self.budget - call_count < 50:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            
            def callback(xk):
                return time.time() < deadline and call_count < self.budget

            try:
                res = minimize(wrapped_func, x0, method='CMA', 
                               options={'maxiter': 1000, 'verbose': 0},
                               callback=callback)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
