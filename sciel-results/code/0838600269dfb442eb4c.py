"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: many L-BFGS-B starts (sorted by center distance) + (1+1)-ES + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

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

        # Phase 1: Many L-BFGS-B starts from diverse initial points
        # Sort by proximity to center since rotated Rosenbrock minimum is often near origin
        n_starts = 15
        starts = []

        # Key structured points
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, -5.0))
        starts.append(np.full(self.dim, 5.0))

        # Random points
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Sort by distance to center (prefer points closer to origin)
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Limit budget per L-BFGS-B start to ensure we try multiple basins
        maxfun_per_start = 80

        for i, x0 in enumerate(starts):
            if not time_ok() or call_count >= self.budget - 80:
                break

            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxls': 30, 'maxfun': maxfun_per_start}
                )
            except Exception:
                pass

            if not time_ok() or call_count >= self.budget - 80:
                break

        # Phase 2: (1+1)-ES refinement around the best point
        if time_ok() and call_count < self.budget - 10:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            stuck = 0
            remaining = self.budget - call_count - 10

            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget - 10:
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

                # Escape mechanism: if stuck too long, restart from best with perturbation
                if stuck > 40:
                    x = np.clip(best_x + 0.3 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.3
                    stuck = 0

        # Phase 3: Final L-BFGS-B polish with tighter bounds around best
        if time_ok() and call_count < self.budget - 5:
            try:
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
                    options={'maxiter': 100, 'maxls': 30, 'maxfun': self.budget - call_count - 5}
                )
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
