# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Powell + L-BFGS-B optimizer.
    Powell's method aligns search directions with the anisotropic valley,
    and L-BFGS-B provides fast local convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                return float(func(x_clipped))
            except Exception:
                return float('inf')

        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Try origin with Powell (small budget) -> L-BFGS-B (large budget)
        # 2. Try random points with Powell (small budget) -> L-BFGS-B (large budget)
        
        n_restarts = 5
        
        for i in range(n_restarts):
            if calls >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            if i == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng.uniform(-5, 5, dim)
            
            # Phase 1: Powell to align axes and get into the valley
            # Use ~30-40 calls for Powell
            powell_budget = min(40, budget - calls - 10)
            if powell_budget < 10:
                break
                
            try:
                res_powell = scipy.optimize.minimize(
                    wrapper, 
                    x_start, 
                    method='Powell',
                    options={
                        'maxiter': 50, 
                        'maxfev': powell_budget
                    }
                )
            except Exception:
                res_powell = None

            x_mid = res_powell.x if res_powell is not None else x_start
            
            # Phase 2: L-BFGS-B to polish
            lbfgs_budget = min(50, budget - calls - 5)
            if lbfgs_budget < 10:
                break

            try:
                res_lbfgs = scipy.optimize.minimize(
                    wrapper, 
                    x_mid, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100, 
                        'maxfev': lbfgs_budget
                    }
                )
                if res_lbfgs.fun < best_f:
                    best_f = res_lbfgs.fun
                    best_x = res_lbfgs.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
