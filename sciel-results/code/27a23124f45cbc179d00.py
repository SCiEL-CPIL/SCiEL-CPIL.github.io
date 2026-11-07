import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        # Track calls
        def tracked_func(x):
            nonlocal calls
            if calls >= budget:
                raise Exception("Budget exceeded")
            if time.time() >= deadline:
                raise Exception("Time limit exceeded")
            calls += 1
            return func(np.clip(x, -5, 5))

        # Phase 1: Multi-start Powell
        # Powell is good for anisotropic functions like Bent Cigar.
        # We use a few restarts to ensure we find a good basin.
        num_restarts = 5
        calls_per_restart = budget // (num_restarts + 1) # Reserve some for polish
        
        for i in range(num_restarts):
            if calls >= budget - 20: # Leave room for polish
                break
            if time.time() >= deadline:
                break

            x0 = rng.uniform(-5, 5, dim)
            # Use a subset of the budget for this restart
            # Powell's maxiter is not exactly calls, but we limit it.
            # We can't easily pass a call counter to minimize, so we rely on maxiter.
            # However, we need to track calls globally.
            # We will run minimize and catch exceptions if it tries to exceed budget.
            
            # To strictly enforce budget, we wrap the function.
            # But `minimize` doesn't stop on exception cleanly in all versions?
            # It does stop and raise.
            
            # We need to be careful: `minimize` might use more calls than `maxiter`?
            # No, `maxiter` limits the number of iterations, but each iteration uses multiple function evaluations.
            # For Powell, one iteration uses dim function evaluations.
            # So maxiter should be roughly calls_per_restart / dim.
            
            max_iter = max(10, calls_per_restart // dim)
            
            try:
                result = minimize(
                    tracked_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': max_iter,
                        'xtol': 1e-10,
                        'ftol': 1e-12
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                # If budget or time exceeded, break
                break

        # Phase 2: Nelder-Mead polish on the best point found
        remaining = budget - calls
        if best_x is not None and remaining > 10 and time.time() < deadline:
            # Nelder-Mead is good for final polishing
            # We need to wrap it to track calls again, but we already have tracked_func
            # However, tracked_func increments `calls` which is shared.
            # This is correct.
            
            try:
                result = minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        return best_x, best_fx
