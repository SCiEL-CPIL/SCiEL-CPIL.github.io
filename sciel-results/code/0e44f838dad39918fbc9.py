"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell's method with dense Latin Hypercube initialization for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        global_best = None
        global_fx = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, global_best, global_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < global_fx:
                global_fx = fx
                global_best = x.copy()
            return fx

        # Phase 1: Dense stratified Latin Hypercube sampling to identify basins
        # Use a larger LHS to better cover the two-basin structure of Lunacek
        n_sample = 120
        lhs_points = []
        for d in range(dim):
            perm = rng.permutation(n_sample)
            points_d = (perm + 0.5) / n_sample * 10.0 - 5.0  # [-5, 5]
            lhs_points.append(points_d)
        lhs_points = np.column_stack(lhs_points)

        # Add targeted points: origin, and points near typical Lunacek minima locations
        extra_points = [np.zeros(dim)]
        # Lunacek often has minima near (0,...,0) and shifted locations
        for _ in range(15):
            extra_points.append(rng.uniform(-4.5, 4.5, dim))

        all_sample_points = np.vstack([lhs_points, np.array(extra_points)])

        # Evaluate all sample points, keeping track of values
        sample_values = []
        for x in all_sample_points:
            if calls_made >= budget - 100 or time.time() > deadline - 40:
                break
            fx = evaluate(x)
            sample_values.append((fx, x.copy()))

        # Sort by function value to get top candidates
        sample_values.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Powell's method from top diverse candidates
        # Select top candidates that are sufficiently diverse
        selected_starts = []
        min_distance = 1.0  # minimum distance between starting points

        # Always include the best point
        if sample_values:
            selected_starts.append(sample_values[0][1].copy())

        # Add diverse points from the sorted list
        for fx, x in sample_values:
            if len(selected_starts) >= 10:
                break
            if time.time() > deadline - 30:
                break
            # Check diversity: ensure this point is far enough from all selected starts
            diverse = True
            for sx in selected_starts:
                if np.linalg.norm(x - sx) < min_distance:
                    diverse = False
                    break
            if diverse:
                selected_starts.append(x.copy())

        # If we didn't get enough diverse points, add some random ones
        while len(selected_starts) < 8 and time.time() < deadline - 30:
            candidate = rng.uniform(-4.5, 4.5, dim)
            diverse = True
            for sx in selected_starts:
                if np.linalg.norm(candidate - sx) < min_distance:
                    diverse = False
                    break
            if diverse:
                selected_starts.append(candidate)

        # Run Powell's method from each selected starting point
        bounds = [(-5.0, 5.0)] * dim

        for x0 in selected_starts:
            if calls_made >= budget - 20 or time.time() > deadline - 15:
                break

            x0 = np.clip(x0, -5.0, 5.0)

            def objective(x):
                if calls_made >= budget - 10 or time.time() > deadline - 10:
                    return np.inf
                return evaluate(x)

            try:
                result = minimize(
                    objective,
                    x0,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'maxfev': 50,
                        'xtol': 1e-8,
                        'ftol': 1e-10,
                    }
                )

                if result.fun < global_fx:
                    global_fx = result.fun
                    global_best = np.clip(result.x, -5.0, 5.0).copy()

            except Exception:
                pass

        # Phase 3: Final polish with L-BFGS-B from the current global best
        if global_best is not None and calls_made < budget - 10 and time.time() < deadline - 10:
            x0 = global_best.copy()

            def objective2(x):
                if calls_made >= budget - 5 or time.time() > deadline - 5:
                    return np.inf
                return evaluate(x)

            try:
                result = minimize(
                    objective2,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if result.fun < global_fx:
                    global_fx = result.fun
                    global_best = np.clip(result.x, -5.0, 5.0).copy()

            except Exception:
                pass

        # Phase 4: Final local search with small perturbations if budget remains
        remaining = budget - calls_made
        if remaining > 5 and global_best is not None and time.time() < deadline - 3:
            x = global_best.copy()
            fx = global_fx
            sigma = 0.05

            for _ in range(min(remaining - 2, 20)):
                if time.time() > deadline - 2:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 0.7
                else:
                    sigma *= 0.85

        return global_best, global_fx
# EVOLVE-BLOCK-END
