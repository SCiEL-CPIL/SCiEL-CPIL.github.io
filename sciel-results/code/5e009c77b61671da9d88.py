"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, fmin, minimize as _min
import time


class Optimizer:
    """Three-phase optimizer: L-BFGS-B fast convergence, COBYLA polish, Nelder-Mead final."""

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

        # Phase 1: L-BFGS-B from center — fast convergence using gradient info
        # Allocate ~60% of budget to L-BFGS-B
        lbf_maxfun = int(max_calls * 0.6)

        x0 = np.zeros(self.dim)
        try:
            result = minimize(
                wrapped_func,
                x0,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={
                    'maxiter': 1000,
                    'maxfun': lbf_maxfun,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
        except Exception:
            if best_x is None:
                best_x = np.zeros(self.dim)
                best_f = wrapped_func(best_x)

        # Phase 2: COBYLA polish — derivative-free, can refine further
        # COBYLA uses finite-difference gradient approximations and can continue
        # improving where L-BFGS-B stopped due to gradient tolerance.
        remaining_after_lbf = max_calls - call_count
        if remaining_after_lbf > 20 and time.time() < deadline - 10:
            cobyla_maxfun = int(remaining_after_lbf * 0.6)
            x_start = best_x.copy()
            try:
                result_cob = minimize(
                    wrapped_func,
                    x_start,
                    method='COBYLA',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 500,
                        'maxfun': cobyla_maxfun,
                        'rhobeg': 0.01,
                        'tol': 1e-14
                    }
                )
            except Exception:
                pass

        # Phase 3: Nelder-Mead final polish with any remaining budget
        remaining_after_cobyla = max_calls - call_count
        if remaining_after_cobyla > 10 and time.time() < deadline - 5:
            x_start = best_x.copy()
            try:
                result_nm = fmin(
                    wrapped_func,
                    x_start,
                    xtol=1e-13,
                    ftol=1e-15,
                    maxiter=200,
                    disp=False
                )
            except Exception:
                pass

        # If no valid result, use a default
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
