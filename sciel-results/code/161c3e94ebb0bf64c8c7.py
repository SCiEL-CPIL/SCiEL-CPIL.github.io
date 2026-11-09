"""Combined optimizer: L-BFGS-B local search with multi-start for Rastrigin rotated f15."""
import numpy as np
import time
import math
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with multi-start for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Initial diverse random sampling
        # Use a mix of uniform and stratified sampling for better coverage
        rng = np.random.default_rng(42)

        n_initial = min(30, budget // 5)
        for i in range(n_initial):
            if not time_ok():
                break
            # Mix of uniform and stratified sampling
            if i < n_initial // 2:
                x = rng.uniform(-5, 5, dim)
            else:
                # Stratified: divide each dimension into bins
                x = np.array([(i % dim) * (10.0 / dim) + rng.uniform(0, 10.0 / dim) - 5.0 for _ in range(dim)])
                x = np.clip(x, -5, 5)
            evaluate(x)

        if not time_ok():
            return (best_x, best_f)

        # Phase 2: L-BFGS-B local search from diverse starting points
        # L-BFGS-B is sample-efficient in low dimensions and can navigate
        # the rotated Rastrigin landscape by exploiting local quadratic structure

        # Generate diverse starting points: best found, perturbed best, and random
        start_points = []

        # Best point found so far
        start_points.append(best_x.copy())

        # Perturbed versions of the best point
        for scale in [0.5, 1.0, 2.0]:
            perturbation = scale * rng.standard_normal(dim)
            x_pert = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_pert)

        # Additional random points
        n_random_starts = 5
        for _ in range(n_random_starts):
            if not time_ok():
                break
            x_rand = rng.uniform(-5, 5, dim)
            start_points.append(x_rand)

        # Sort starting points by their function values (evaluate if not already done)
        # We'll run L-BFGS-B from each, but prioritize better starting points
        start_points.sort(key=lambda x: best_f if np.array_equal(x, best_x) else 1e10)

        # Budget allocation: give more budget to better starting points
        remaining_budget = budget - calls_used
        n_starts = len(start_points)

        # Allocate budget: top starts get more evaluations
        # Use a geometric allocation
        weights = np.array([1.0 / (i + 1) for i in range(n_starts)])
        weights /= weights.sum()
        evals_per_start = [max(10, int(w * remaining_budget)) for w in weights]
        # Cap each start to not exceed remaining budget
        total_allocated = sum(evals_per_start)
        if total_allocated > remaining_budget:
            scale_factor = remaining_budget / total_allocated
            evals_per_start = [int(e * scale_factor) for e in evals_per_start]

        for idx, x_start in enumerate(start_points):
            if not time_ok():
                break

            max_evals = evals_per_start[idx]
            if max_evals < 5:
                continue

            # Run L-BFGS-B with the given starting point
            # Use a callback to track evaluations and stop if budget exceeded
            eval_count = [0]

            def objective(x):
                eval_count[0] += 1
                if not time_ok() or eval_count[0] > max_evals:
                    return 1e10  # Large value to discourage further search
                return evaluate(x)

            try:
                result = minimize(
                    objective,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max_evals, 'maxfun': max_evals}
                )
            except Exception:
                continue

            # Update best if L-BFGS-B found something better
            # The evaluate function already tracks best_x and best_f

        # Phase 3: Final refinement with L-BFGS-B from the best solution
        if time_ok() and budget - calls_used > 10:
            max_evals = min(50, budget - calls_used)

            def refine_objective(x):
                if not time_ok():
                    return 1e10
                return evaluate(x)

            try:
                result = minimize(
                    refine_objective,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': max_evals, 'maxfun': max_evals, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, do a few more random restarts with L-BFGS-B
        while time_ok() and budget - calls_used > 20:
            x_rand = rng.uniform(-5, 5, dim)
            fx_rand = evaluate(x_rand)

            if fx_rand < best_f + 10:  # Only refine if reasonably close
                max_evals = min(30, budget - calls_used)

                def restart_objective(x):
                    if not time_ok():
                        return 1e10
                    return evaluate(x)

                try:
                    minimize(
                        restart_objective,
                        x_rand,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': max_evals, 'maxfun': max_evals}
                    )
                except Exception:
                    pass

        return (best_x, best_f)
