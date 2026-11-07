"""Optimizer for the 5-function BBOB-style suite task."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B for smooth BBOB functions, with a coarse
    global sweep for multimodal cases and a strict wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0   # safety margin before the 300 s limit
        dim = self.dim
        budget = self.budget

        # Deterministic, well-spread starting points (no hidden seed needed).
        def lattice_points(n):
            pts = []
            if n <= 0:
                return pts
            coords = np.linspace(-4.0, 4.0, n)
            for v in np.ndindex((n,) * dim):
                pts.append(np.array([coords[i] for i in v], dtype=float))
            return pts

        # A few extra random points to help multimodal / shifted instances.
        rng = np.random.default_rng(12345)
        extra = [rng.uniform(-4.0, 4.0, dim) for _ in range(12)]

        starts = lattice_points(5) + extra

        best_x = None
        best_f = np.inf

        def consider(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Run L-BFGS-B from each start while budget and time allow.
        bounds = [(-5.0, 5.0)] * dim
        used = 0
        for s in starts:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                res = minimize(func, s, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-15,
                                        'gtol': 1e-12, 'maxls': 50})
            except Exception:
                continue
            used += res.nfev
            f = float(res.fun)
            consider(res.x, f)

        # If we still have budget left (cheap instances), do a light global
        # refinement sweep with small steps from the best point.
        if used < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            for step in (1.0, 0.3, 0.1, 0.03):
                for d in range(dim):
                    if used >= budget or time.time() >= deadline:
                        break
                    for sign in (1.0, -1.0):
                        if used >= budget or time.time() >= deadline:
                            break
                        y = x.copy()
                        y[d] = np.clip(y[d] + sign * step, -5.0, 5.0)
                        fy = float(func(y))
                        used += 1
                        if fy < fx:
                            x, fx = y, fy
            consider(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
