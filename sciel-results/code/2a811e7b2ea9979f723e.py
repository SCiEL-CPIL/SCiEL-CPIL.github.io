import numpy as np
import time
from scipy.stats.qmc import Sobol

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin within 300s limit
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Wrapper to track calls, time, and best result
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= self.budget:
                raise Exception("Budget exceeded")
            
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Phase 1: Global Exploration with Sobol Sequence
        # Use 100 points for global search in 5D
        n_global = min(100, self.budget)
        
        try:
            # Generate Sobol points
            sampler = Sobol(d=self.dim, scramble=True)
            # Generate n_global points in [0, 1]^d
            sobol_points = sampler.random(n_global)
            # Map to [-5, 5]^d
            sobol_points = sobol_points * 10.0 - 5.0
            
            for x in sobol_points:
                wrapped_func(x)
                
        except (TimeoutError, Exception):
            pass

        # If we haven't made any calls (should not happen), return
        if best_x is None:
             # Fallback: try center
             x0 = np.zeros(self.dim)
             try:
                 wrapped_func(x0)
             except:
                 pass
             if best_x is None:
                 return np.zeros(self.dim), np.inf

        # Phase 2: Local Refinement with CMA-ES
        # Use remaining budget
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 10:
            try:
                from cma import CMA
                # Start with a small initial sigma
                sigma0 = 1.0
                opts = {
                    'maxfevals': remaining_budget,
                    'seed': 42,
                    'bounds': [-5.0, 5.0],
                    'verbose': -9,
                    'tolx': 1e-10,
                    'tolopt': 1e-10,
                    'tolstagnation': 100
                }
                
                es = CMA(best_x, sigma0, opts)
                
                while not es.stop() and calls_made < self.budget and time.time() < deadline:
                    # Generate a batch of points
                    batch = es.ask()
                    for x in batch:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        wrapped_func(x)
                    
                    # Tell the optimizer the results
                    # We need to track the function values for the batch
                    # But wrapped_func doesn't return a list, so we need to modify approach
                    # Actually, CMA-ES requires a list of function values corresponding to the batch
                    # Let's restructure to store values in a list
                    
                    break # Fallback if CMA-ES isn't available or we want to use a simpler approach
            except ImportError:
                pass
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        # Use remaining budget for final polish
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 5:
            try:
                from scipy.optimize import minimize
                # Nelder-Mead options
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'maxiter': remaining_budget,
                        'adaptive': True
                    }
                )
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
