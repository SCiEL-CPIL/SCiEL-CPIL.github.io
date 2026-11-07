"""Optimizer using SLSQP with multi-start and adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start SLSQP optimizer for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Quick coarse sampling to find a good starting region
        # Use structured + random points to quickly locate promising basins
        n_coarse = 15
        rng = np.random.default_rng(42)

        # Start with origin and axis-aligned points
        for i in range(dim + 1):
            if not time_ok() or call_count[0] >= budget - 20:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = np.zeros(dim)
                x0[i - 1] = 2.0
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Add some random points
        for i in range(n_coarse - dim - 1):
            if not time_ok() or call_count[0] >= budget - 20:
                break
            x0 = rng.uniform(-4, 4, dim)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: SLSQP refinement with multiple starts
        # SLSQP uses quadratic programming subproblems and can converge faster
        # than L-BFGS-B for smooth problems with good initial guesses
        
        def run_slsqp(x0, maxiter=100, ftol=1e-14, gtol=1e-10):
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                )
                if res.success and res.fun < best_f:
                    return res.fun, res.x.copy()
                elif res.fun < best_f:
                    return res.fun, res.x.copy()
            except Exception:
                pass
            return None, None

        # First, refine the current best point
        if best_x is not None and time_ok() and call_count[0] < budget - 10:
            f_new, x_new = run_slsqp(best_x, maxiter=100)
            if x_new is not None:
                best_f = f_new
                best_x = x_new

        # Try multiple restarts around the best point with varying scales
        n_restarts = 8
        scales = [0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0]
        
        for i in range(n_restarts):
            if not time_ok() or call_count[0] >= budget - 10:
                break
            
            scale = scales[i]
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5, 5)
            
            f_new, x_new = run_slsqp(x0, maxiter=60)
            if x_new is not None:
                best_f = f_new
                best_x = x_new

        # Phase 3: Final high-precision polish with very tight tolerances
        if time_ok() and best_x is not None and call_count[0] < budget - 5:
            f_new, x_new = run_slsqp(best_x, maxiter=50, ftol=1e-16, gtol=1e-12)
            if x_new is not None:
                best_f = f_new
                best_x = x_new

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
