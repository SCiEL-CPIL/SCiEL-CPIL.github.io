"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: L-BFGS-B from multiple starts + CMA-ES-like refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multiple L-BFGS-B starts from diverse initial points
        # Use a set of well-spread starting points
        n_starts = 15
        starts = []

        # Origin and box corners
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, -5.0))
        starts.append(np.full(self.dim, 5.0))

        # Random points
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Sort by a heuristic: prefer points closer to center (Rosenbrock minimum is often near origin after rotation)
        def center_dist(x):
            return np.linalg.norm(x)

        starts.sort(key=center_dist)

        for i, x0 in enumerate(starts):
            if not time_ok() or call_count >= self.budget - 50:
                break

            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxls': 50, 'maxfun': 500}
                )
                if res.success and res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            if not time_ok() or call_count >= self.budget - 50:
                break

        # Phase 2: Local refinement with a simple (1+1)-ES around the best point
        if time_ok() and call_count < self.budget - 10:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            stuck = 0
            remaining = self.budget - call_count - 5

            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stuck = 0
                else:
                    sigma *= 0.9
                    stuck += 1

                if stuck > 50:
                    # Restart from a slightly perturbed best
                    x = np.clip(best_x + 0.3 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.3
                    stuck = 0

        # Final: try a few more L-BFGS-B from the best point with tighter bounds
        if time_ok() and call_count < self.budget - 10:
            try:
                # Narrow bounds around best
                bounds = []
                for i in range(self.dim):
                    lo = max(-5, best_x[i] - 1.0)
                    hi = min(5, best_x[i] + 1.0)
                    bounds.append((lo, hi))

                res = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxls': 30, 'maxfun': 200}
                )
                if res.success and res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
