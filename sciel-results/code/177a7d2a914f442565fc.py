"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task using L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with finite-difference gradients and adaptive polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Wrap func to count calls and handle clipping
        def count_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return 0.0
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        def run_lbfgsb(x0, max_iter=1000, tol=1e-10, ftol=1e-12, gtol=1e-8):
            """Run L-BFGS-B from x0."""
            nonlocal calls, best_x, best_f
            
            if calls >= budget or time.time() > deadline:
                return x0, best_f
            
            # Use finite differences for gradient
            # eps for forward difference
            eps = 1e-6
            
            try:
                result = minimize(
                    count_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': max_iter,
                        'ftol': ftol,
                        'gtol': gtol,
                        'maxls': 20,
                        'maxfun': 200  # Limit function evals per minimize call
                    }
                )
            except Exception:
                pass
            
            # Update best if result is better
            if result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5, 5).copy()
            
            return best_x, best_f

        # Reserve 20% of budget for polishing
        polish_budget = int(budget * 0.2)
        search_budget = budget - polish_budget

        # First start at 0 - good heuristic for rotated Rosenbrock
        x_start = np.zeros(n)
        f_start = count_func(x_start)
        
        if calls >= search_budget:
            # Go straight to polish
            run_lbfgsb(best_x, max_iter=200, tol=1e-12, ftol=1e-14, gtol=1e-10)
            return best_x, best_f

        # Run L-BFGS-B from 0
        x_curr, f_curr = run_lbfgsb(x_start, max_iter=500, tol=1e-10, ftol=1e-12, gtol=1e-8)
        
        num_restarts_done = 1
        max_restarts = 12
        
        # Adaptive restart: continue while we're making progress and have budget
        for restart_idx in range(max_restarts - 1):
            if calls >= search_budget or time.time() > deadline:
                break
            
            # Generate a new random starting point
            x_new = rng.uniform(-5, 5, n)
            f_new = count_func(x_new)
            
            if calls >= search_budget:
                break

            # Run L-BFGS-B from this point
            x_curr, f_curr = run_lbfgsb(x_new, max_iter=300, tol=1e-10, ftol=1e-12, gtol=1e-8)
            num_restarts_done += 1
            
            # If we've converged well, we can stop exploring
            if best_f < 1e-4:
                if restart_idx >= 5:
                    break

        # High-precision polish pass on the best solution found
        if calls < budget and time.time() < deadline:
            # Use very tight tolerances for polish
            run_lbfgsb(best_x, max_iter=100, tol=1e-14, ftol=1e-16, gtol=1e-12)
            
            # One more polish with even tighter settings
            if calls < budget and time.time() < deadline:
                run_lbfgsb(best_x, max_iter=50, tol=1e-16, ftol=1e-18, gtol=1e-14)

        return best_x, best_f
# EVOLVE-BLOCK-END
