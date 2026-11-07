"""Rosenbrock-style f8 optimizer with coordinate descent + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Rosenbrock-style f8 on [-5,5]^5.
    Strategy:
      1. Multi-start sampling with structured + random points.
      2. Coordinate descent (cycling through dimensions) with line search.
      3. Nelder-Mead simplex polish for final convergence.
      4. If budget remains, additional local refinement cycles.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = [0]
        budget = self.budget

        def wrapped(x):
            if call_count[0] >= budget:
                return best_f if best_f != np.inf else 0.0
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e10
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Multi-start scouting
        # Use structured points (corners, edges, center) plus random points
        n_scouts = min(30, budget // 3)
        scout_points = []

        # Center
        scout_points.append(np.zeros(self.dim))

        # Axis-aligned points: each axis at -2, 0, 2 (but not all zeros)
        for d in range(self.dim):
            for val in [-2.0, 0.0, 2.0]:
                pt = np.zeros(self.dim)
                pt[d] = val
                if np.any(pt != 0):
                    scout_points.append(pt)

        # Random points
        n_random = max(0, n_scouts - len(scout_points))
        for _ in range(n_random):
            scout_points.append(rng.uniform(-4.0, 4.0, self.dim))

        # Evaluate all scout points
        for x0 in scout_points:
            if call_count[0] >= budget:
                break
            if time.time() > deadline - 120:
                break
            x0 = np.clip(x0, -5, 5)
            fx0 = wrapped(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            x0 = rng.uniform(-5, 5, self.dim)
            best_f = wrapped(x0)
            best_x = x0.copy()

        # Phase 2: Coordinate descent with line search
        # Cycle through dimensions, for each do a 1D search
        if call_count[0] < budget and time.time() < deadline - 60:
            x = best_x.copy()
            f = best_f
            for iteration in range(200):
                if call_count[0] >= budget or time.time() > deadline - 60:
                    break
                improved = False
                for d in range(self.dim):
                    if call_count[0] >= budget or time.time() > deadline - 60:
                        break
                    # 1D search along dimension d
                    best_1d = f
                    best_val = x[d]
                    # Try several step sizes
                    for step in [0.5, 0.25, 0.1, 0.05, -0.5, -0.25, -0.1, -0.05]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + step, -5.0, 5.0)
                        f_new = wrapped(x_new)
                        if f_new < best_1d:
                            best_1d = f_new
                            best_val = x_new[d]
                    if best_val != x[d]:
                        x[d] = best_val
                        f = best_1d
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                if not improved:
                    break

        # Phase 3: Nelder-Mead polish
        if call_count[0] < budget and time.time() < deadline - 15:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True, 'maxfev': budget - call_count[0]}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If budget remains, do more coordinate descent from current best
        if call_count[0] < budget and time.time() < deadline - 5:
            x = best_x.copy()
            f = best_f
            for iteration in range(100):
                if call_count[0] >= budget or time.time() > deadline - 5:
                    break
                improved = False
                for d in range(self.dim):
                    if call_count[0] >= budget or time.time() > deadline - 5:
                        break
                    best_1d = f
                    best_val = x[d]
                    for step in [0.1, 0.05, 0.02, 0.01, -0.1, -0.05, -0.02, -0.01]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x[d] + step, -5.0, 5.0)
                        f_new = wrapped(x_new)
                        if f_new < best_1d:
                            best_1d = f_new
                            best_val = x_new[d]
                    if best_val != x[d]:
                        x[d] = best_val
                        f = best_1d
                        improved = True
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
