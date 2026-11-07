"""Rewritten program for BBOB f9 (Rosenbrock rotated) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style f9 (Rosenbrock rotated).
    
    Strategy:
    1. Quick global exploration: evaluate a set of diverse starting points.
    2. Local exploitation: run L-BFGS-B from the best few starting points.
    3. Refinement: a final L-BFGS-B pass with tight bounds around the best solution.
    
    The budget is strictly tracked via a wrapper around func.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global exploration with diverse starting points ---
        # Generate a set of candidate starting points.
        # Include origin, axis-aligned points, and random points.
        starts = []
        starts.append(np.zeros(self.dim))
        
        # Axis-aligned points: +/- 1, +/- 2, +/- 3 along each axis
        for i in range(self.dim):
            for val in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
                pt = np.zeros(self.dim)
                pt[i] = val
                starts.append(pt)
        
        # Random points
        n_random = 30
        for _ in range(n_random):
            starts.append(rng.uniform(-5.0, 5.0, self.dim))

        # Evaluate all starting points to find the best few
        start_evals = []
        for x0 in starts:
            if call_count >= self.budget - 100:  # Reserve budget for local optimization
                break
            if not time_ok():
                break
            x0_clipped = np.clip(x0, -5.0, 5.0)
            fx0 = tracked_func(x0_clipped)
            start_evals.append((fx0, x0_clipped))

        # Sort by function value and take the top few
        start_evals.sort(key=lambda t: t[0])
        top_starts = start_evals[:5]  # Take top 5 starting points

        # --- Phase 2: Local optimization from top starts ---
        # Run L-BFGS-B from each of the top starts
        for fx0, x0 in top_starts:
            if call_count >= self.budget - 50:
                break
            if not time_ok():
                break
            
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxls': 50,
                        'maxfun': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 3: Final refinement with tight bounds ---
        if call_count < self.budget - 10 and time_ok():
            try:
                # Narrow bounds around the best solution
                bounds = []
                for i in range(self.dim):
                    lo = max(-5.0, best_x[i] - 0.5)
                    hi = min(5.0, best_x[i] + 0.5)
                    bounds.append((lo, hi))

                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxls': 30,
                        'maxfun': 50,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Additional refinement if budget remains ---
        if call_count < self.budget - 5 and time_ok():
            try:
                # Even tighter bounds
                bounds = []
                for i in range(self.dim):
                    lo = max(-5.0, best_x[i] - 0.1)
                    hi = min(5.0, best_x[i] + 0.1)
                    bounds.append((lo, hi))

                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxls': 20,
                        'maxfun': 20,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
