"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin before 330s kill
        
        n = self.dim
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # We will perform multiple restarts.
        # Each restart uses a Nelder-Mead simplex.
        # We need to track calls carefully to not exceed budget.
        
        # Strategy:
        # 1. Generate a set of starting points.
        # 2. For each starting point, run Nelder-Mead until it converges or budget/time runs out.
        # 3. Keep the best result.
        
        # How many restarts?
        # Nelder-Mead is efficient. Let's try 10-20 restarts.
        # If dim=5, a simplex has 6 points.
        # Initial evaluations per restart: ~6.
        # Steps per restart: ~100-200.
        # Total budget 500.
        # Let's aim for ~10 restarts, each getting ~50 calls? Or just run until convergence.
        
        # Better: Run NM, check calls. If it converges early, restart.
        # If it doesn't converge, stop and try another start.
        
        # We need a wrapper to count calls.
        def counting_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return best_f # Return best known to stop optimization? No, NM might get stuck.
                             # Actually, if we hit budget, we should stop the loop.
            calls_made += 1
            return func(x)
            
        # Pre-generate a few diverse starting points to avoid wasting time on RNG if we need to restart quickly
        # But NM needs to be run sequentially.
        
        # Let's use a fixed set of deterministic-ish starts + random ones
        # Or just random starts.
        
        # To maximize AOCC, we want fast initial progress.
        # NM usually converges quickly to a local minimum.
        # f4 is multi-modal.
        
        # Let's try 20 restarts.
        # If we run out of budget or time, we stop.
        
        for i in range(30): # Max 30 restarts
            if time.time() - start_time > timeout:
                break
            if calls_made >= self.budget:
                break
                
            # Initialize simplex
            # Start point
            x0 = np.random.uniform(-5, 5, n)
            
            # Options for Nelder-Mead
            # maxiter controls the number of iterations, not function evaluations directly.
            # We need to limit function evaluations.
            # We can use a custom callback or just check calls_made in the loop if we implement it ourselves.
            # SciPy's minimize with Nelder-Mead doesn't have a direct 'maxfev' option in all versions?
            # Actually, `maxiter` is iterations. Each iteration is ~1 function eval.
            # Let's use maxiter=50 per restart.
            
            # Wait, if I use scipy.optimize.minimize, I can't easily interrupt it mid-iteration if it's stuck in a loop.
            # But 50 iterations is fast.
            
            # Let's refine: 
            # Use scipy.optimize.minimize with method='Nelder-Mead'
            # options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
            
            # But we must ensure we don't exceed budget.
            # If calls_made + 50 > self.budget, we should reduce maxiter or skip.
            
            remaining = self.budget - calls_made
            if remaining <= 0:
                break
                
            max_iter_for_this = min(50, remaining)
            
            # If max_iter_for_this is small, it's not worth restarting?
            # NM needs at least n+1 evals to start.
            if max_iter_for_this < n + 1:
                break
                
            # Define the function to minimize
            # We need to be careful: if counting_func returns best_f when budget is hit, 
            # NM might think it found a minimum and stop. That's fine.
            
            res = minimize(counting_func, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter_for_this, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
                
            # Check if we should restart
            # If it converged (res.success) or hit maxiter, we restart.
            # Nelder-Mead often hits maxiter without "success" if not fully converged.
            
            # Optional: If the improvement was tiny, maybe try a more distant start?
            # For now, just random restarts.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
