"""Multi-restart SciPy optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart optimizer for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Generate diverse starting points tailored for Rosenbrock valley
        rng = np.random.default_rng(42)
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Points along diagonal (common valley direction for Rosenbrock)
        for scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
            starts.append(np.full(dim, scale))
        # Random points in [-4, 4]
        for _ in range(15):
            starts.append(rng.uniform(-4, 4, dim))
        # Points with one coordinate offset (axis-aligned exploration)
        for i in range(dim):
            for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        # Some points near boundaries
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 4.0
            starts.append(pt)
            pt[i] = -4.0
            starts.append(pt)

        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # Adaptive restart strategy
        # Start with moderate maxfev, increase if making progress
        base_maxfev = 30
        min_maxfev = 10
        max_maxfev = 80
        current_maxfev = base_maxfev

        for x0 in starts:
            if time.time() > deadline or call_count >= budget:
                break
            
            # Skip if we've already found a very good solution and this start is far
            if best_f < 1.0 and best_x is not None:
                dist = np.linalg.norm(x0 - best_x)
                if dist > 4.0:
                    continue

            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 150, 'maxfev': current_maxfev}
                )
                if res.fun < best_f:
                    # Made progress, allow more budget for next restart
                    current_maxfev = min(max_maxfev, int(current_maxfev * 1.3))
                    update_best(res.x, res.fun)
                else:
                    # No improvement, reduce budget for next restart
                    current_maxfev = max(min_maxfev, int(current_maxfev * 0.7))
            except Exception:
                pass

        # Final local refinement from the best point with remaining budget
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 5:
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 200, 'maxfev': remaining}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # Second refinement pass if budget remains
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={'maxiter': 200, 'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
