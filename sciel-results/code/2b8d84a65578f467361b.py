"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based Optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Optimization with CMA-ES
        # Population size 10 is good for dim 5.
        # Max iterations to stay within budget: 500 // 10 = 50.
        # We add a small buffer.
        popsize = 10
        max_iter_1 = budget // popsize
        
        x0 = np.zeros(dim)
        
        try:
            res = minimize(
                safe_call, 
                x0, 
                method='CMA-ES',
                options={
                    'maxiter': max_iter_1,
                    'popsize': popsize,
                    'sigma0': 1.0,
                    'tolfun': 1e-15,
                    'tolx': 1e-15
                }
            )
        except Exception:
            pass

        # Phase 2: Refinement if budget remains
        # If CMA-ES converged early or used less than budget, refine.
        if calls_made < budget - 20 and time.time() < deadline:
            # Use a smaller population and sigma for refinement
            popsize_2 = 8
            remaining_budget = budget - calls_made
            max_iter_2 = remaining_budget // popsize_2
            
            if max_iter_2 > 5: # Only run if we have enough for a few iterations
                try:
                    res2 = minimize(
                        safe_call, 
                        best_x, 
                        method='CMA-ES',
                        options={
                            'maxiter': max_iter_2,
                            'popsize': popsize_2,
                            'sigma0': 0.1, # Smaller step
                            'tolfun': 1e-15,
                            'tolx': 1e-15
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
