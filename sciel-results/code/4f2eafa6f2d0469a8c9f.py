"""Optimizer using multi-start L-BFGS-B with adaptive restarts for Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer with adaptive restarts for box-constrained minimization on [-5,5]^5."""

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
            if call_count[0] >= budget:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            if np.isnan(val) or np.isinf(val):
                return 1e10
            return val

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count[0] < budget

        # Generate initial points: mix of structured and random
        rng = np.random.RandomState(42)
        candidates = []

        # Origin
        candidates.append(np.zeros(dim))

        # Points along axes
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                candidates.append(x)

        # Random points in the box
        for _ in range(10):
            x = rng.uniform(-4, 4, size=dim)
            candidates.append(x)

        # Points near the typical Rosenbrock minimum region (shifted by hidden transform)
        # The Rosenbrock function has its minimum at (1,1,...,1) in standard form.
        # With hidden shift/rotation, we don't know exactly where, but we can try
        # several points that might be closer to the minimum.
        for _ in range(5):
            x = rng.uniform(-2, 2, size=dim)
            candidates.append(x)

        # Shuffle candidates
        rng.shuffle(candidates)

        # Multi-start L-BFGS-B with adaptive restarts
        for x0 in candidates:
            if not (time_ok() and budget_ok()):
                break

            try:
                # Run L-BFGS-B from this starting point
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

            except Exception:
                # If L-BFGS-B fails, try Nelder-Mead as fallback
                try:
                    res_nm = minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass

            # If we've made good progress, try to polish with SLSQP
            if best_x is not None and time_ok() and budget_ok():
                try:
                    res_slsqp = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 100, 'ftol': 1e-14}
                    )
                    if res_slsqp.fun < best_f:
                        best_f = res_slsqp.fun
                        best_x = res_slsqp.x.copy()
                except Exception:
                    pass

            # If we've used more than 80% of budget, stop
            if call_count[0] > budget * 0.8:
                break

        # Final polish if we have a best point and time remains
        if best_x is not None and time_ok() and budget_ok():
            try:
                res_final = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except Exception:
                pass

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if budget_ok():
                try:
                    best_f = wrapped_func(best_x)
                except Exception:
                    best_f = 0.0
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
