"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, fmin
import time


class Optimizer:
    """Single high-precision L-BFGS-B + Nelder-Mead polish optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit
        
        # Count function calls
        call_count = 0
        max_calls = self.budget
        
        # Best solution found
        best_x = None
        best_f = np.inf
        
        # Wrap func to count calls, enforce budget, and track best
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                return 1e10  # Return large value to stop optimization
            call_count += 1
            # Clip to box
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: L-BFGS-B from center with tight tolerances
        # Discus is unimodal, so starting from center is optimal.
        # Allocate ~80% of budget to L-BFGS-B.
        lbf_maxfun = int(max_calls * 0.8)
        
        x0 = np.zeros(self.dim)
        try:
            # Use minimize with L-BFGS-B
            # Tight tolerances to push convergence as far as possible
            result = minimize(
                wrapped_func,
                x0,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={
                    'maxiter': 1000, 
                    'maxfun': lbf_maxfun,
                    'ftol': 1e-15,
                    'gtol': 1e-12
                }
            )
            # best_x and best_f are updated in wrapped_func
        except Exception:
            # Fallback if L-BFGS-B fails
            if best_x is None:
                best_x = np.zeros(self.dim)
                best_f = wrapped_func(best_x)

        # Phase 2: Nelder-Mead polish with remaining budget
        # Use remaining calls to polish the local minimum
        remaining_calls = max_calls - call_count
        if remaining_calls > 10 and time.time() < deadline - 5:
            # Nelder-Mead is derivative-free, good for final polish
            # Initialize simplex with small spread
            x_start = best_x.copy()
            
            try:
                # fmin uses Nelder-Mead
                # We rely on wrapped_func to stop when budget is hit (returns 1e10)
                # and to track the best point.
                # maxiter is set high, but budget cap will effectively limit it.
                result_nm = fmin(
                    wrapped_func,
                    x_start,
                    xtol=1e-12,
                    ftol=1e-15,
                    maxiter=200,
                    disp=False
                )
                # best_x and best_f are updated in wrapped_func
            except Exception:
                pass
        
        # If no valid result, use a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
