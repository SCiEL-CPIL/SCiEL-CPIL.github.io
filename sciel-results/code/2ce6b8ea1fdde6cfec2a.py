"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol multi-start with L-BFGS-B for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        # Wrap func to track calls
        call_count = [0]

        def tracked_func(x):
            x = np.asarray(x, dtype=float)
            if x.shape != (d,):
                x = x.reshape(d)
            result = func(x)
            call_count[0] += 1
            return result

        best_x = None
        best_f = np.inf

        # Use Sobol sequence for initial diverse points
        from scipy.stats import qmc

        # Generate Sobol points in [0,1]^d, then map to [-5,5]^d
        # We want enough points to cover the space well
        # For d=5, use about 64-128 initial points
        n_sobol = min(128, budget - 50)  # Leave some budget for refinement
        sobol = qmc.Sobol(d=d, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        # Map from [0,1] to [-5,5]
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate Sobol points
        for i in range(n_sobol):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            x = sobol_points[i]
            x = np.clip(x, -5, 5)
            f = tracked_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            # Fallback: evaluate center
            x = np.zeros(d)
            f = tracked_func(x)
            best_x = x.copy()
            best_f = f

        # Now do sequential refinement on the best candidates
        # Use L-BFGS-B with bounds
        bounds = [(-5.0, 5.0)] * d

        # Try refining from the best point found so far
        if call_count[0] < budget and time.time() < deadline:
            try:
                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': budget - call_count[0], 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we still have budget, try refining from a few other good starting points
        # Re-evaluate some Sobol points to find other local minima candidates
        # Actually, let's try a few more starting points for multi-start
        # Use the next batch of Sobol points
        n_more = min(32, budget - call_count[0] - 50)
        if n_more > 0 and time.time() < deadline:
            sobol_points2 = sobol.random(n_more)
            sobol_points2 = sobol_points2 * 10.0 - 5.0

            for i in range(n_more):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                x = sobol_points2[i]
                x = np.clip(x, -5, 5)
                f = tracked_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Refine from the new best if it improved
            if call_count[0] < budget and time.time() < deadline:
                try:
                    res = minimize(
                        tracked_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfun': budget - call_count[0], 'gtol': 1e-8}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Final check: make sure we don't exceed budget
        # The return value is ignored by the evaluator, but keep it for interface
        return best_x, best_f
# EVOLVE-BLOCK-END
