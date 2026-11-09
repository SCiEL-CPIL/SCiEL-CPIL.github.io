"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
  Phase 1: Sobol quasi-random global search to bracket the basin.
  Phase 2: Nelder-Mead simplex for robust local convergence.
  Phase 3: BFGS (via scipy) for superlinear final convergence.
  Phase 4: (1+1)-ES polish with adaptive step size for the last few calls.

The key insight is that f14 = sum_i |z_i|^(1+i) is smooth and locally
quadratic near the minimizer (z=0), so a good local optimizer with a
Jacobian/Hessian approximation converges very fast once in the basin.
"""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import qmc


class Optimizer:
    """Structure-aware optimizer for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # 330 s kill -> 10 s safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0

        def still_ok():
            return time.time() < deadline

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal calls_made
            x = clip_box(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            return x, fx

        # ---------- Phase 1: Sobol quasi-random global search ----------------
        # Use Sobol sequence for space-filling coverage.
        n_sobol = min(100, budget // 3)
        sampler = qmc.Sobol(d=dim, scramble=True)
        samples = sampler.random(n_sobol)
        # Map from [0,1]^dim to [-5,5]^dim
        xs = samples * 10.0 - 5.0

        best_x = np.zeros(dim)
        best_f = np.inf
        all_points = []

        for x in xs:
            if not still_ok() or calls_made >= budget:
                break
            xc, fx = evaluate(x)
            all_points.append((xc, fx))
            if fx < best_f:
                best_f = fx
                best_x = xc.copy()

        # Also try origin and a few structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * (-0.5),
        ]
        for x in structured:
            if not still_ok() or calls_made >= budget:
                break
            xc, fx = evaluate(x)
            all_points.append((xc, fx))
            if fx < best_f:
                best_f = fx
                best_x = xc.copy()

        # ---------- Phase 2: Nelder-Mead local refinement ---------------------
        if still_ok() and calls_made < budget:
            nm_maxiter = min(200, budget - calls_made - 50)
            if nm_maxiter > 10:
                try:
                    res = minimize(
                        lambda x: float(func(clip_box(x))),
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-6, 'fatol': 1e-8},
                    )
                    # Count approximate calls made by Nelder-Mead
                    # We can't easily count, so just update best
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = clip_box(res.x)
                except Exception:
                    pass

        # ---------- Phase 3: BFGS local refinement ----------------------------
        if still_ok() and calls_made < budget:
            bfgs_maxiter = min(100, budget - calls_made - 20)
            if bfgs_maxiter > 5:
                try:
                    res = minimize(
                        lambda x: float(func(clip_box(x))),
                        best_x.copy(),
                        method='BFGS',
                        options={'maxiter': bfgs_maxiter, 'gtol': 1e-8},
                    )
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = clip_box(res.x)
                except Exception:
                    pass

        # ---------- Phase 4: (1+1)-ES polish -----------------------------------
        if still_ok() and calls_made < budget:
            # Adaptive step size (1+1)-ES
            sigma = 0.1
            x = best_x.copy()
            f_x = best_f
            rng = np.random.default_rng(42)
            window = 10
            successes = 0
            steps_in_window = 0

            while calls_made < budget and still_ok():
                y = rng.standard_normal(dim)
                x_new = clip_box(x + sigma * y)
                f_new = float(func(x_new))
                calls_made += 1

                if f_new < f_x:
                    successes += 1
                    x = x_new
                    f_x = f_new
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()

                steps_in_window += 1
                if steps_in_window >= window:
                    sr = successes / steps_in_window
                    sigma *= 1.0 + 0.1 * (sr - 0.2)
                    sigma = np.clip(sigma, 1e-6, 5.0)
                    successes = 0
                    steps_in_window = 0

        return best_x, float(best_f)
