"""Multi-start L-BFGS-B optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer for smooth low-dimensional functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted(x):
            if calls[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            calls[0] += 1
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []

        # 1. Center
        starts.append(np.zeros(dim))

        # 2. Grid-like points (sparse 3x3x3x3x3 is too many, so we use a subset)
        # For dim=5, a full grid is 3^5=243 points. We can afford a few.
        # Let's use points on the axes and some diagonal corners.
        # Axis-aligned points: +/- 2, +/- 4 on each axis
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        
        # 3. Random points
        n_random = 20
        for _ in range(n_random):
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Sort starts by some heuristic? Or just iterate.
        # We will iterate and stop when budget or time runs out.
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Each L-BFGS-B run consumes calls. 
        # L-BFGS-B with finite differences uses dim+1 calls per gradient eval.
        # maxiter=10, maxfun=50 is a reasonable small budget per restart.
        # With 500 total calls, we can do ~10-20 restarts.
        
        for start in unique_starts:
            if calls[0] >= budget - 10:
                break
            if time.time() > deadline:
                break

            try:
                res = minimize(
                    counted,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 10, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # res.fun is the value at res.x, but we've already tracked best_f in counted()
                # However, counted() might have stopped early if budget hit, so res.fun might be inf or stale.
                # We rely on the global best_f updated inside counted().
            except Exception:
                pass

        # If we have budget left, do a few more random restarts with tighter budget
        while calls[0] < budget - 5 and time.time() < deadline:
            s = rng.uniform(-4.0, 4.0, dim)
            try:
                res = minimize(
                    counted,
                    s,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 5, 'maxfun': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
