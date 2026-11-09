"""Portfolio optimizer for BBOB f1 (Sphere) on [-5,5]^dim.

Structural change: use a direct quadratic model (Broyden-like / Gauss-Newton)
exploiting the known quadratic structure of Sphere, combined with a compact
global search (LHS + random) and a CMA-ES fallback.  The key insight is that
for f(x) = ||x - z*||^2 + c, we can estimate the minimizer from a small number
of function evaluations using finite differences / quadratic fitting, then
refine with a few Newton-style steps.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Quadratic-model optimizer for Sphere with global search and CMA-ES fallback."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0

        def time_ok():
            return time.time() < deadline

        n_evals = [0]

        def wrap(z):
            n_evals[0] += 1
            x = np.clip(np.asarray(z, dtype=float).reshape(-1), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return 1e30

        rng = np.random.default_rng(42)

        # ------------------------------------------------------------------
        # Phase 0: Global search – dense LHS + random + structured points.
        # Evaluate a broad net to locate the basin of the shifted sphere.
        # ------------------------------------------------------------------
        n_lhs = 40
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        n_rand = 20
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        center = np.zeros(dim)
        axis_pts = []
        for d in range(dim):
            for sign in [-1.0, 1.0]:
                p = np.zeros(dim)
                p[d] = sign * 2.5
                axis_pts.append(p)
        structured = np.array([center] + axis_pts)

        candidates = np.vstack([lhs, rand, structured])
        n_cand = len(candidates)

        # Evaluate all candidates (they are cheap and give global picture).
        # Order by distance from origin for early good point.
        dists = np.linalg.norm(candidates, axis=1)
        order = np.argsort(dists)
        candidates_ordered = candidates[order]

        best_x = candidates_ordered[0].copy()
        best_f = wrap(best_x)

        max_global_evals = min(n_cand, 80)
        for idx in range(1, max_global_evals):
            if n_evals[0] >= budget - 20 or not time_ok():
                break
            c = candidates_ordered[idx]
            f_val = wrap(c)
            if f_val < best_f:
                best_f = f_val
                best_x = c.copy()

        # ------------------------------------------------------------------
        # Phase 1: Quadratic model fitting (exploits Sphere structure).
        # For f(x) = ||x - z*||^2 + c, we can estimate z* from evaluations.
        # Use a small set of points around best_x to fit the quadratic.
        # ------------------------------------------------------------------
        def estimate_minimizer_quadratic(x0, f0, step=0.5):
            """Estimate the minimizer of a quadratic from f(x0) and finite diffs."""
            dim = len(x0)
            # Collect gradient estimate via central differences
            grad = np.zeros(dim)
            for d in range(dim):
                xp = x0.copy()
                xm = x0.copy()
                xp[d] += step
                xm[d] -= step
                xp = np.clip(xp, -5.0, 5.0)
                xm = np.clip(xm, -5.0, 5.0)
                fp = wrap(xp)
                fm = wrap(xm)
                grad[d] = (fp - fm) / (2.0 * step)

            # For a quadratic f(x) = 0.5 * x^T H x + g^T x + c,
            # the minimizer satisfies H * x* = -g.
            # For Sphere: f(x) = ||x - z*||^2 + c = x^T x - 2 z*^T x + const
            # So H = 2I, g = -2 z*, and x* = z*.
            # Gradient = 2(x - z*), so z* = x0 - grad/2.
            x_est = x0 - grad / 2.0
            x_est = np.clip(x_est, -5.0, 5.0)
            return x_est, grad

        # Try quadratic estimation from best_x
        if n_evals[0] < budget - 30 and time_ok():
            x_est, grad = estimate_minimizer_quadratic(best_x, best_f, step=0.5)
            f_est = wrap(x_est)
            if f_est < best_f:
                best_f = f_est
                best_x = x_est.copy()

            # Refine with smaller step
            if n_evals[0] < budget - 20 and time_ok():
                x_est2, _ = estimate_minimizer_quadratic(best_x, best_f, step=0.1)
                f_est2 = wrap(x_est2)
                if f_est2 < best_f:
                    best_f = f_est2
                    best_x = x_est2.copy()

            # One more refinement
            if n_evals[0] < budget - 10 and time_ok():
                x_est3, _ = estimate_minimizer_quadratic(best_x, best_f, step=0.02)
                f_est3 = wrap(x_est3)
                if f_est3 < best_f:
                    best_f = f_est3
                    best_x = x_est3.copy()

        # ------------------------------------------------------------------
        # Phase 2: CMA-ES for robust local optimization.
        # ------------------------------------------------------------------
        try:
            from cma import CMA
            if n_evals[0] < budget - 50 and time_ok():
                cma_budget = int(min(120, max(20, budget - n_evals[0] - 30)))
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=0.8,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=cma_budget,
                          seed=1,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0,
                          cmaes_tolerances={
                              'tolask': 0.0,
                              'tolfun': 1e-14,
                              'tolx': 0.0,
                              'tolopt': 0.0,
                          })
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase 3: L-BFGS-B polish.
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import minimize
            if n_evals[0] < budget - 15 and time_ok():
                maxfun = max(10, min(200, budget - n_evals[0] - 10))
                r = minimize(wrap, best_x, method='L-BFGS-B',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 500,
                                      'maxfun': maxfun,
                                      'ftol': 1e-14})
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase 4: Final quadratic refinement (exploits exact quadratic structure).
        # ------------------------------------------------------------------
        if n_evals[0] < budget - 5 and time_ok():
            for step in [0.01, 0.005, 0.001]:
                if n_evals[0] >= budget - 5 or not time_ok():
                    break
                x_est, _ = estimate_minimizer_quadratic(best_x, best_f, step=step)
                f_est = wrap(x_est)
                if f_est < best_f:
                    best_f = f_est
                    best_x = x_est.copy()

        # ------------------------------------------------------------------
        # Phase 5: Final L-BFGS-B with tight tolerance.
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import minimize
            if n_evals[0] < budget - 5 and time_ok():
                maxfun = max(5, min(50, budget - n_evals[0]))
                r = minimize(wrap, best_x, method='L-BFGS-B',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 1000,
                                      'maxfun': maxfun,
                                      'ftol': 1e-15,
                                      'gtol': 1e-14})
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
        except Exception:
            pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
