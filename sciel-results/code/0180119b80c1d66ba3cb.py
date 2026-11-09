"""Multi-start Nelder-Mead portfolio for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin for 300s limit
        
        # Track global best
        best_x = None
        best_f = float('inf')
        
        # Budget allocation: 5 starts, 100 evals each = 500 total
        n_starts = 5
        evals_per_start = self.budget // n_starts
        
        # Deterministic seed for reproducibility if needed, but rng helps explore
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            # Check time
            if time.time() > deadline:
                break
                
            # Initialize start point
            # Mix of center and random to cover space
            if i == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)
                
            # Clip to box
            x0 = np.clip(x0, -5, 5)
            
            # Run Nelder-Mead
            # maxiter is an upper bound, but we also rely on time and evals
            # Nelder-Mead in scipy uses 'maxiter' as the number of iterations (function evals are roughly maxiter * (dim+1) in worst case, 
            # but usually less. To be safe with budget, we set maxiter such that 
            # it doesn't blow past the local budget. 
            # Actually, scipy's Nelder-Mead maxiter is the number of iterations of the simplex.
            # Each iteration does 1-2 evals. So maxiter=50 is ~50-100 evals.
            # Let's use a tight maxiter to respect the 100 eval budget per start.
            local_max_iter = int(evals_per_start * 0.8) # Conservative
            
            try:
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': local_max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if res.success or (res.fun < best_f):
                    # Clip result to ensure validity, though NM usually stays close
                    x_res = np.clip(res.x, -5, 5)
                    # Verify function value at clipped point if it changed? 
                    # The task says func clips. So res.fun is from unclipped or clipped?
                    # "func(x) clips a finite length-5 vector to the box and returns a float."
                    # So func handles clipping. res.fun is valid.
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
            except Exception:
                # If NM crashes or fails, ignore and move to next start
                pass
                
        # If no valid result found (unlikely), return center
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
