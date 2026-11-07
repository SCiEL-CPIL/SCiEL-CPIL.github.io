"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5, 5]^5.

Strategy (structural change from CMA-ES to multi-start (1+1)-ES + Nelder-Mead):
  * Front-load a coarse global grid to establish a decent best-so-far quickly.
  * Run multi-start Nelder-Mead simplex from diverse starting points to exploit
    the rotated landscape structure.
  * Use a (1+1)-ES with adaptive step size for robust local refinement with
    explicit escape mechanisms (restarts/jumps) to prevent stagnation.
  * Final L-BFGS-B polish on the best point found.
  * Explicit wall-clock deadline guard with a safety margin.
"""
import time
import numpy as np
from scipy.optimize import minimize as _sp_min


class Optimizer:
    """Continuous black-box minimizer for f15 (Rastrigin rotated) on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 330 s hard kill -> 50 s safety margin
        dim = self.dim
        budget = self.budget

        def _left():
            return time.time() < deadline

        # ---- state ----
        x_best = np.zeros(dim)
        f_best = np.inf
        calls = 0

        def _wrap(x):
            nonlocal calls
            calls += 1
            return func(x)

        def _update(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # ---- 1) cheap global grid: front-load early progress ----
        # 3^5 = 243 points, step 2.5. For Rastrigin with shift in [-4,4], a
        # coarse grid reliably lands near a local minimum, giving a strong
        # early b_i and a solid seed for further search.
        grid_levels = 3
        gvals = np.linspace(-5.0, 5.0, grid_levels)
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        for i4 in range(grid_levels):
                            if not _left():
                                break
                            x = np.array([gvals[i0], gvals[i1], gvals[i2],
                                          gvals[i3], gvals[i4]], dtype=float)
                            _update(x, _wrap(x))
                        if not _left():
                            break
                    if not _left():
                        break
                if not _left():
                    break
            if not _left():
                break
        if not _left():
            return x_best, f_best

        # ---- 2) Multi-start Nelder-Mead simplex ----
        # Start from the best grid point and several random points near it.
        rng = np.random.default_rng(42)
        nm_starts = [x_best.copy()]
        # Add a few random points in the box
        for _ in range(4):
            x_rand = rng.uniform(-5.0, 5.0, size=dim)
            nm_starts.append(x_rand)
        # Add points near the best
        for _ in range(4):
            x_near = x_best + rng.normal(0, 1.0, size=dim)
            x_near = np.clip(x_near, -5.0, 5.0)
            nm_starts.append(x_near)

        for x0 in nm_starts:
            if not _left():
                break
            try:
                res = _sp_min(_wrap, x0, method="Nelder-Mead",
                              options={"maxiter": 50, "xatol": 1e-6, "fatol": 1e-6})
                if res.x is not None:
                    _update(res.x, float(res.fun))
            except Exception:
                pass
        if not _left():
            return x_best, f_best

        # ---- 3) (1+1)-ES with adaptive step size and escape mechanisms ----
        # This is the main robust workhorse. We run multiple restarts, each
        # seeded from the current global best, with adaptive sigma and
        # explicit jumps to escape local minima.
        max_sigma = 4.0
        min_sigma = 0.01
        num_restarts = 8
        es_budget_per_restart = max(10, (budget - calls) // num_restarts)

        for restart in range(num_restarts):
            if not _left():
                break
            x = x_best.copy()
            f = f_best
            sigma = min(2.0, max_sigma)
            rng_es = np.random.default_rng(12345 + restart)
            no_improve = 0
            max_no_improve = 20

            for step in range(es_budget_per_restart):
                if not _left():
                    break
                # Generate candidate
                y = x + sigma * rng_es.normal(0, 1, size=dim)
                y = np.clip(y, -5.0, 5.0)
                fy = _wrap(y)

                if fy < f:
                    # Accept
                    x = y
                    f = fy
                    _update(x, f)
                    sigma = max(sigma * 1.2, min_sigma)
                    no_improve = 0
                else:
                    # Reject
                    sigma = max(sigma * 0.8, min_sigma)
                    no_improve += 1
                    # Escape mechanism: if stuck, jump
                    if no_improve >= max_no_improve:
                        x = x_best + rng_es.normal(0, 1.5, size=dim)
                        x = np.clip(x, -5.0, 5.0)
                        f = _wrap(x)
                        _update(x, f)
                        sigma = 2.0
                        no_improve = 0

        if not _left():
            return x_best, f_best

        # ---- 4) Final polish with L-BFGS-B ----
        try:
            res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                          bounds=[(-5.0, 5.0)] * dim,
                          options={"maxiter": 40, "maxls": 20})
            if res.x is not None:
                _update(res.x, float(res.fun))
        except Exception:
            pass

        return x_best, f_best
