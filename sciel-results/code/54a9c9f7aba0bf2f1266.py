"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0
        time_exceeded = False

        def wrapped_func(x):
            nonlocal call_count, time_exceeded
            if call_count >= budget or time.time() > deadline:
                time_exceeded = True
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return fx
            except Exception:
                return np.inf

        # Multi-start strategy
        # We use 10 restarts. Each gets a portion of the budget.
        n_restarts = 10
        evals_per_restart = budget // n_restarts
        if evals_per_restart < 10:
            n_restarts = max(1, budget // 10)
            evals_per_restart = budget // n_restarts

        for i in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break
            
            # Random start point
            x0 = rng.uniform(-5, 5, d)
            
            # Initial evaluation
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            if time_exceeded or call_count >= budget:
                break

            # Run Nelder-Mead
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_restart, 
                                        'xatol': 1e-5, 
                                        'fatol': 1e-5,
                                        'adaptive': True})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                        
            except Exception:
                pass

            if time_exceeded or call_count >= budget:
                break

        # Fallback
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
