import numpy as np
import time
from scipy.optimize import minimize

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        # Safety margin: 300s limit, kill at 330s. Aim to finish by 300s.
        deadline = t0 + 290
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        max_evals = self.budget

        def wrapper(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= max_evals:
                # Should not happen if cma respects budget, but safe guard
                return best_f
            eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        if HAS_CMA:
            x0 = np.zeros(self.dim)
            sigma0 = 1.0
            # Popsize 10 is reasonable for dim 5. 500/10 = 50 generations.
            options = {
                'maxiter': 100, # Generations limit
                'popsize': 10,
                'verbose': -9,
                'seed': 0,
                'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                'CMA_adapt_sigma': 'default',
                'CMA_boundaryhandler': 'reflect'
            }
            
            try:
                # cma.fmin returns [x, f, es, log]
                res = cma.fmin(wrapper, x0, sigma0, options=options)
                if res is not None and len(res) > 0:
                    # Update best if cma found something better (wrapper already tracks, but just in case)
                    pass 
            except Exception as e:
                # If CMA fails, we fall through to fallback or return what we have
                pass

        # If CMA-ES was not used or failed to make progress, run a fallback ES
        # This ensures we use the budget if CMA stops early or is missing
        if not HAS_CMA or eval_count < max_evals * 0.5:
            # Fallback: Simple (1+lambda)-ES or similar
            # If we already used most of the budget, don't restart
            if eval_count < max_evals:
                rng = np.random.default_rng(42)
                # Start from current best or random
                if best_x is not None:
                    x = best_x.copy()
                else:
                    x = rng.uniform(-5, 5, self.dim)
                    fx = wrapper(x) # Uses 1 eval
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                sigma = 0.5
                # Run for remaining budget
                while eval_count < max_evals and time.time() < deadline:
                    # Generate 1 step
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                    fy = wrapper(y)
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        x = y.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                        # Simple restart logic if stuck
                        # Keep it simple to avoid complexity bugs
                        # If no improvement for a while, maybe re-center? 
                        # For now, just let it run.
        
        # Ensure we return a valid array
        if best_x is None:
            best_x = np.zeros(self.dim)
            # If we never evaluated, evaluate once? 
            # The problem says "exception before the first call makes result invalid".
            # We must make at least 1 call.
            if eval_count == 0:
                try:
                    best_f = float(func(best_x))
                    eval_count = 1
                except Exception:
                    best_f = float('inf')
            
        return best_x, best_f
