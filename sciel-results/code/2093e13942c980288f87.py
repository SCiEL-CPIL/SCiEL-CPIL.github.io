"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead for Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        
        # We have 500 function evaluations total.
        # Nelder-Mead in 5D typically uses ~20-50 evaluations per restart to converge decently.
        # Let's aim for about 10-15 restarts to balance exploration and exploitation.
        # Each restart will have a limited maxiter.
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Strategy: 
        # 1. First, do a quick random sampling to find a decent starting point? 
        #    Or just start with several random points.
        # Given the "Bent Cigar" nature, the valley is narrow. 
        # NM is good at following valleys if the initial simplex is not too skewed.
        
        # Let's allocate the budget.
        # If we do 10 restarts, we have 50 evaluations per restart.
        # scipy.optimize.minimize with method='Nelder-Mead' has maxiter and maxfev.
        # maxfev controls function calls.
        
        num_restarts = 10
        calls_per_restart = self.budget // num_restarts
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Generate initial point
            # To improve chances, we might want to cluster starts or spread them.
            # Uniform random is standard.
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Wrap the function to handle clipping and count calls?
            # The task says func(x) clips. We just pass it.
            # However, we need to ensure we don't exceed the global budget.
            # scipy's maxfev is per call to minimize.
            
            try:
                # Use Nelder-Mead
                # initial_simplex is important. Default is standard basis scaled by x0.
                # For Bent Cigar, a larger initial simplex might help find the valley direction faster?
                # Or smaller to be precise?
                # Let's stick to default or slightly larger.
                
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': calls_per_restart, 'maxfev': calls_per_restart, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                # If an exception occurs, we keep the calls made so far (handled by evaluator)
                # but we should stop trying to restart if we are in trouble.
                # The task says: "an exception later keeps the calls made".
                # So if minimize throws, we just break.
                break

        if best_x is None:
             # Fallback if no successful restart (unlikely but possible if first one fails immediately)
             best_x = np.zeros(self.dim)
             try:
                 best_f = func(best_x)
             except:
                 pass

        return best_x, best_f
# EVOLVE-BLOCK-END
