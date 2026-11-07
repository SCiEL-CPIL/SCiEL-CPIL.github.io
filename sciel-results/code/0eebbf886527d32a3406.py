# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Uses L-BFGS-B optimization with random restarts, suitable for the
    Bent Cigar function's anisotropic landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0  # safety margin
        dim = self.dim
        budget = self.budget
        calls = 0

        # Wrapper to count calls and handle any potential issues
        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                return float(func(x))
            except Exception:
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Try multiple random starts with L-BFGS-B
        rng = np.random.default_rng(42)
        
        # Number of restarts: balance exploration and local optimization
        # Each L-BFGS-B run uses ~50-100 evaluations typically
        n_restarts = 8
        
        for i in range(n_restarts):
            if calls >= budget - 10:
                break
            if time.time() > deadline:
                break
            
            # Generate random start point
            if i == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng.uniform(-5, 5, dim)
            
            try:
                result = scipy.optimize.minimize(
                    wrapper, 
                    x_start, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': min(100, budget - calls - 5)}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
