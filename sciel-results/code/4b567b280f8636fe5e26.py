"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to count calls and check deadline
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= self.budget or time.time() > deadline - 1:
                # Safety check: if we exceed budget, raise to stop.
                # But we must ensure this never happens in valid runs.
                raise RuntimeError("Budget exceeded")
            
            calls_made += 1
            return func(x)

        # Phase 1: Multi-start exploration
        # 5 restarts, each with maxiter=8
        # Estimated calls: 5 * (6 * (1+8)) = 270
        
        n_restarts = 5
        maxiter_restart = 8
        
        for i in range(n_restarts):
            if time.time() > deadline - 10: break
            
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                res = minimize(counted_func, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter_restart, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except RuntimeError:
                break

        # Phase 2: Exploit the best found
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5:
            remaining_budget = self.budget - calls_made
            # Estimated calls per maxiter: 6
            # Total calls = 6 * (1 + maxiter)
            # So maxiter = (remaining_budget // 6) - 1
            maxiter_final = (remaining_budget // 6) - 1
            
            if maxiter_final > 0:
                try:
                    res = minimize(counted_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter_final, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except RuntimeError:
                    pass
        
        # If no calls were made (should not happen), do a single random call
        if calls_made == 0:
            x0 = rng.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
