"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with deterministic starting points."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track the best solution found so far
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Deterministic multi-start points covering the box [-5, 5]^5
        # The minimizer is in [-4, 4]^5
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0])
        ]
        
        # Allocate budget across restarts
        # First start gets the most, as it's the most likely to be close or converge fast
        # Subsequent starts get less
        # We use 5 starts, but budget might run out.
        # 40%, 25%, 20%, 10%, 5%
        budget_per_restart = [0.4, 0.25, 0.2, 0.1, 0.05]
        
        for i, start_x in enumerate(starts):
            if calls_made >= budget:
                break
            
            remaining = budget - calls_made
            alloc = int(remaining * budget_per_restart[i])
            # Ensure we have enough for at least one iteration of NM (dim+1 calls)
            if alloc < dim + 2:
                alloc = remaining
            
            if alloc <= 0:
                break

            # Nelder-Mead options
            # maxiter is the max number of iterations.
            # Each iteration typically consumes 1 function call (after initial simplex setup).
            # Initial simplex setup consumes dim + 1 calls.
            # So max_iter = alloc - (dim + 1)
            max_iter = max(1, alloc - (dim + 1))
            
            # Set initial simplex size. 
            # For the first start (0), a step of 1.0 is good.
            # For others, maybe smaller to stay in bounds? 
            # scipy default is based on x0. If x0 is 0, it uses 0.05 * dim? 
            # Let's explicitly set initial_simplex for control.
            # If start is 0, simplex points are 0 +/- step.
            # If start is 2, simplex points are 2 +/- step.
            
            # We'll let scipy handle it, but tune xatol/fatol for speed.
            try:
                res = minimize(
                    evaluate, 
                    start_x, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
