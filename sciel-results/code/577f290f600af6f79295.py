"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using Random Sampling followed by Powell's Method with adaptive restarts.
    Combines the structural exploration of multi-start with the refinement power of Powell.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard variables
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin from 330s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            # Check deadline
            if time.time() > deadline:
                return float('inf')
            
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Random Exploration
        # Use a fraction of the budget for exploration to find a good starting point
        # 10 random points is a good balance for dim=5 to avoid wasting budget
        n_explore = min(10, self.budget // 10)
        
        for _ in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # Phase 2: Local Exploitation using Powell's Method
        # Powell is effective for Rosenbrock as it adapts search directions
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            # Add a small buffer to ensure we don't exceed budget due to scipy overhead
            # though maxfev should be strict
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={
                        'maxfev': remaining,
                        'xtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
                # Update best if minimize found something better
                # wrapped_func already updates best_f, so best_f is current best
            except Exception:
                pass

        # Phase 3: If we still have significant budget and time, try another local search
        # from a different random point if the first one didn't converge well
        # This helps if the first Powell run got stuck in a local minimum or converged prematurely
        if calls < self.budget - 50 and time.time() < deadline - 10 and best_x is not None:
            # Check if we made significant progress (heuristic: if best_f is still very high, maybe we didn't find the valley)
            # For Rosenbrock, f=0 is optimal. If we're still > 1, we might be far away.
            # But we don't know f_opt exactly. Let's just try another short Powell from a new random point
            # if we have plenty of budget left.
            
            # Pick a new random point
            x_new = rng.uniform(-5, 5, self.dim)
            f_new = wrapped_func(x_new)
            
            if f_new < best_f * 0.5 or (best_f > 10 and f_new < best_f):
                # Only continue if this new point is promising
                remaining = self.budget - calls
                if remaining > 20:
                    try:
                        res = minimize(
                            wrapped_func, 
                            x_new, 
                            method='Powell', 
                            options={
                                'maxfev': remaining,
                                'xtol': 1e-10,
                                'ftol': 1e-10
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
