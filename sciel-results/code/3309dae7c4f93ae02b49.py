"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse initialization for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Counter for function calls
        call_count = [0]
        max_calls = budget

        def safe_call(x):
            if call_count[0] >= max_calls:
                return float('inf')
            call_count[0] += 1
            try:
                val = func(x)
                if not np.isfinite(val):
                    return float('inf')
                return float(val)
            except Exception:
                return float('inf')

        # Track best solution found
        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Diverse initialization - mix of structured and random points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        start_points = []

        # Center
        start_points.append(np.zeros(dim))

        # Axis-aligned points (structured)
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 3.5
                start_points.append(pt)

        # Corner points
        for corner in [(-4, -4, -4, -4, -4), (4, 4, 4, 4, 4),
                       (-4, 4, -4, 4, -4), (4, -4, 4, -4, 4),
                       (-4, -4, 4, 4, -4), (4, 4, -4, -4, 4)]:
            start_points.append(np.array(corner[:dim], dtype=float))

        # Random points
        n_random = 30
        for _ in range(n_random):
            pt = rng.uniform(-4.5, 4.5, dim)
            start_points.append(pt)

        # Sobol-like quasi-random points for better coverage
        # Use a simple Latin hypercube
        n_lhs = 20
        for _ in range(n_lhs):
            pt = np.zeros(dim)
            for i in range(dim):
                pt[i] = (rng.random() * 9 - 4.5)
            start_points.append(pt)

        # Evaluate all start points
        for pt in start_points:
            if call_count[0] >= max_calls or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            f = safe_call(pt_clipped)
            update_best(pt_clipped, f)

        # Phase 2: Local optimization from best points
        # Sort start points by their function values and run L-BFGS-B from best ones
        # First, collect all evaluated points with their values
        evaluated = []
        for pt in start_points:
            if call_count[0] >= max_calls or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            # We already evaluated these, but let's track them
            # Actually, we need to re-evaluate or track properly
            pass

        # Re-do: collect evaluated points
        evaluated_points = []
        for pt in start_points:
            if call_count[0] >= max_calls or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            f = safe_call(pt_clipped)
            evaluated_points.append((f, pt_clipped))

        # Sort by function value
        evaluated_points.sort(key=lambda x: x[0])

        # Run L-BFGS-B from the top N starting points
        n_local = min(15, len(evaluated_points))
        bounds = [(-5, 5)] * dim

        for i in range(n_local):
            if call_count[0] >= max_calls or time.time() > deadline:
                break

            f0, x0 = evaluated_points[i]

            # L-BFGS-B optimization
            try:
                result = minimize(
                    safe_call,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(10, (max_calls - call_count[0]) // max(1, n_local - i)),
                        'gtol': 1e-6
                    }
                )
                if result.fun < best_f:
                    update_best(np.clip(result.x, -5, 5), result.fun)
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more random restarts with local opt
        if call_count[0] < max_calls and time.time() < deadline:
            for _ in range(10):
                if call_count[0] >= max_calls or time.time() > deadline:
                    break
                x0 = rng.uniform(-4.5, 4.5, dim)
                f0 = safe_call(x0)
                update_best(x0, f0)

                try:
                    result = minimize(
                        safe_call,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'maxfun': max(5, (max_calls - call_count[0]) // 10),
                            'gtol': 1e-6
                        }
                    )
                    if result.fun < best_f:
                        update_best(np.clip(result.x, -5, 5), result.fun)
                except Exception:
                    pass

        # Phase 4: Final polish - if we have budget left, do a fine local search around best
        if call_count[0] < max_calls and time.time() < deadline and best_x is not None:
            try:
                result = minimize(
                    safe_call,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': max(10, max_calls - call_count[0]),
                        'gtol': 1e-8
                    }
                )
                if result.fun < best_f:
                    update_best(np.clip(result.x, -5, 5), result.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
