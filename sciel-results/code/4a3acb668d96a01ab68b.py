"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio optimizer: random sampling + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def tracked_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Random sampling to find promising regions
        n_random = min(50, self.budget // 10)
        candidates = []
        for i in range(n_random):
            if calls_made >= self.budget - 50:
                break
            if time.time() > deadline - 30:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = tracked_func(x)
            candidates.append((fx, x.copy()))
        
        # Sort by fitness and pick top K for local refinement
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        
        # Phase 2: L-BFGS-B from each top candidate
        for i in range(top_k):
            if calls_made >= self.budget - 10:
                break
            if time.time() > deadline - 10:
                break
            
            x0 = candidates[i][1]
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 20, 'maxls': 5}
                )
            except Exception:
                continue
        
        return best_x, best_f
# EVOLVE-BLOCK-END
