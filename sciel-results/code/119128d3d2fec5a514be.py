"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.optimize import minimize
import time

class Optimizer:
    """Grid search followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # 1. Global Phase: Grid Search
        # The minimum is in [-4, 4]^5.
        # Use a grid of 3 points per dimension: -4, 0, 4.
        # Total points: 3^5 = 243.
        grid_vals = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(grid_vals, repeat=self.dim))
        
        results = []
        for p in grid_points:
            if time.time() > deadline:
                break
            x = np.array(p, dtype=float)
            try:
                fx = func(x)
                results.append((fx, x))
            except Exception:
                pass
        
        if not results:
            # Fallback to random if grid failed
            rng = np.random.default_rng(0)
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx

        # Sort by function value
        results.sort(key=lambda t: t[0])
        
        # Select top K candidates for local refinement
        # K=5 to cover multiple potential basins
        K = 5
        best_starts = [x for fx, x in results[:K]]
        
        best_x = best_starts[0]
        best_fx = results[0][0]
        
        # 2. Local Phase: Refinement
        # Use Nelder-Mead from each of the top starts.
        # Limit iterations to stay within budget.
        # Nelder-Mead uses ~1 eval per iteration after initial simplex.
        # Initial simplex: dim + 1 = 6 evals.
        # maxiter=40 -> ~46 evals per start.
        # 5 starts * 46 = 230 evals.
        # Total ~ 243 + 230 = 473 < 500.
        
        for x0 in best_starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 40, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                continue
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
