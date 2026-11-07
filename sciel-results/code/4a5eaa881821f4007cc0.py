"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with Nelder-Mead polish for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        bounds = [(-5.0, 5.0)] * n
        
        # Phase 1: Multi-start L-BFGS-B
        # We use a number of starts that fits within the budget.
        # Each start uses a small local budget.
        # Let's say we try 20 starts, each with maxfun=10. Total ~200 calls.
        # Then we have ~300 calls left for polishing.
        
        n_starts = 20
        maxfun_per_start = 15
        
        for i in range(n_starts):
            if time.time() > deadline or calls_used >= budget - 100:
                break
            
            # Generate initial point
            # Mix of random and slightly structured to cover the box
            if i % 2 == 0:
                x0 = rng.uniform(-4.5, 4.5, n)
            else:
                # Try points closer to the origin or corners
                x0 = rng.uniform(-2.0, 2.0, n)
            
            try:
                res = minimize(
                    func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'maxfun': maxfun_per_start, 'gtol': 1e-5}
                )
                calls_used += res.nfev
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 2: Polish the best point found
        if best_x is not None:
            # Use remaining budget for a high-quality local optimization
            remaining_budget = budget - calls_used
            
            # First, try L-BFGS-B with a larger budget
            if remaining_budget > 50 and time.time() < deadline:
                try:
                    res2 = minimize(
                        func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': min(remaining_budget - 20, 200), 'gtol': 1e-8}
                    )
                    calls_used += res2.nfev
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()
                except Exception:
                    pass
            
            # Second, try Nelder-Mead for final polish if budget allows
            remaining_budget = budget - calls_used
            if remaining_budget > 20 and time.time() < deadline:
                try:
                    res3 = minimize(
                        func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'maxfev': min(remaining_budget, 100), 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    calls_used += res3.nfev
                    if res3.fun < best_f:
                        best_f = res3.fun
                        best_x = res3.x.copy()
                except Exception:
                    pass

        # If no valid best_x was found (should not happen), fallback
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
