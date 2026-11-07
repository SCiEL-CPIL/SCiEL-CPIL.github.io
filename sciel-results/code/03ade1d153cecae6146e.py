"""Rewritten optimizer: Multi-start Powell with (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import powell


class Optimizer:
    """Multi-start Powell's method with (1+1)-ES polish and a hard wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0  # 5 s safety margin before the 300 s eval limit

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        # ---------- Phase 1: coarse random sampling to locate the basin ----------
        # Use 15 starts to ensure we find the basin in [-4, 4]^5
        n_starts = 15
        candidates = rng.uniform(-4.0, 4.0, size=(n_starts, dim))
        # Include origin and a few structured points
        extra = np.array([
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0
        ])
        candidates = np.vstack([candidates, extra])
        
        start_xs = []
        start_fs = []
        for i in range(len(candidates)):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = clip(candidates[i])
            f = func(x)
            evaluated += 1
            start_xs.append(x)
            start_fs.append(f)
            if f < best_f:
                best_f, best_x = f, x.copy()

        # ---------- Phase 2: Powell's Method from the best start ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            # Run Powell's method
            # maxfev ensures we don't exceed budget
            # ftol/xtol set tight to encourage convergence
            try:
                # We pass the raw func, but we need to track calls.
                # SciPy's powell handles the calls internally and counts them in res.nfev
                res = powell(func, best_x, maxiter=remaining, maxfev=remaining, ftol=1e-12, xtol=1e-12)
                evaluated += res.nfev
                if res.fun < best_f:
                    best_f, best_x = res.fun, res.x.copy()
            except Exception:
                # If Powell fails, we just proceed with the best found so far
                pass

        # ---------- Phase 3: final fine polish with (1+1)-ES ----------
        remaining = budget - evaluated
        if remaining > 0 and best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_f
            # Start with a small sigma for polish
            sigma = 0.01
            successes = 0
            window = 0
            WIN = 5
            for _ in range(remaining):
                if not elapsed_ok() or evaluated >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                evaluated += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                if window >= WIN:
                    if successes / window > 1.0 / 5.0:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-6, 1.0))
                    successes = 0
                    window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
