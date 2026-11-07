"""Multi-start Nelder-Mead with random restarts for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        n_calls = [0]
        
        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Use Latin Hypercube Sampling for initial points to ensure coverage
        # We will perform multiple Nelder-Mead runs.
        # Each NM run typically consumes ~10-50 calls depending on convergence.
        # We allocate a portion of the budget to each restart.
        
        # Strategy: 
        # 1. Generate K random starting points.
        # 2. For each point, run Nelder-Mead with a limited budget.
        # 3. Keep the best result.
        
        # To maximize coverage with 500 calls in dim=5:
        # Let's try 5 restarts.
        # Initial simplex for NM uses dim+1 = 6 points.
        # If we allow 100 calls per restart, that's plenty for NM to converge or stall.
        
        rng = np.random.default_rng(42)
        
        # Number of restarts
        n_restarts = 5
        calls_per_restart = budget // n_restarts
        
        for i in range(n_restarts):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            # Generate a starting point
            # Use a mix of center and random
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4.0, 4.0, size=dim)
            
            # Wrapper for this specific restart to track calls and stop
            restart_calls = [0]
            def nm_func(x):
                if restart_calls[0] >= calls_per_restart or n_calls[0] >= budget or time.time() > deadline:
                    return np.inf
                restart_calls[0] += 1
                return safe_func(x)
            
            try:
                res = minimize(
                    nm_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, # Limit iterations to prevent infinite loops
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Final polish with remaining budget if any
        remaining = budget - n_calls[0]
        if remaining > 10 and time.time() < deadline:
            polish_calls = [0]
            def polish_func(x):
                if polish_calls[0] >= remaining or n_calls[0] >= budget or time.time() > deadline:
                    return np.inf
                polish_calls[0] += 1
                return safe_func(x)
            
            try:
                res = minimize(
                    polish_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
