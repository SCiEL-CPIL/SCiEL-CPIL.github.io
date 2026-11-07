"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with time guard for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Track best result
        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f if np.isfinite(best_f) else 0.0
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            return val

        # Generate multi-start points: corners, center, and random
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners of the box
        for corner in [[-5.0]*dim, [5.0]*dim]:
            starts.append(np.array(corner))
        # Mid-edge points
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = -5.0
            starts.append(pt.copy())
            pt[i] = 5.0
            starts.append(pt.copy())
        # Random points
        n_random = max(1, min(10, (budget - len(starts)) // 50))
        for _ in range(n_random):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle starts for diversity
        rng.shuffle(starts)

        # Allocate budget across starts
        # Give more budget to earlier starts (they're more likely to be good)
        n_starts = len(starts)
        # Use a decay: first start gets more budget
        base_budget = max(10, budget // (n_starts * 3))
        budgets = []
        remaining = budget
        for i in range(n_starts):
            if i == 0:
                b = min(remaining, max(50, budget // 4))
            elif i < n_starts // 2:
                b = min(remaining, max(20, budget // 8))
            else:
                b = min(remaining, max(10, budget // 16))
            budgets.append(b)
            remaining -= b
            if remaining <= 0:
                break

        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if call_count[0] >= budget:
                break

            # Run L-BFGS-B from this start
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxls': 50,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': max(5, budget - call_count[0]),
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

            # If we've made good progress, do a local refinement
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            # Additional local search with smaller step
            if best_x is not None and call_count[0] < budget:
                try:
                    res2 = minimize(
                        wrapped,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': 50,
                            'maxls': 30,
                            'gtol': 1e-8,
                            'ftol': 1e-12,
                            'maxfun': max(5, budget - call_count[0]),
                        }
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Final check
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
