"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: (1+1)-ES with 1/5 success rule and restarts
        es_budget = int(budget * 0.75)
        
        sigma = 2.0
        no_improve = 0
        no_improve_limit = 30
        restarts = 0
        max_restarts = 5
        
        for _ in range(es_budget - 1):
            if not time_ok() or calls >= es_budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                no_improve = 0
                sigma = min(sigma * 1.25, 4.0)
            else:
                no_improve += 1
                sigma = max(sigma * 0.85, 0.05)
                
                if no_improve >= no_improve_limit and restarts < max_restarts:
                    x = best_x + sigma * 0.5 * rng.standard_normal(d)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if calls >= es_budget:
                        break
                    no_improve = 0
                    sigma = max(1.0, sigma * 1.5)
                    restarts += 1

        # Phase 2: Nelder-Mead polish from best point with call tracking
        if time_ok() and calls < budget:
            try:
                from scipy.optimize import minimize
                remaining = budget - calls
                if remaining > 5:
                    call_count = [calls]
                    
                    def counted_func(v):
                        call_count[0] += 1
                        if call_count[0] >= budget:
                            return 1e30
                        return func(np.clip(v, -5, 5))
                    
                    res = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    calls = call_count[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
