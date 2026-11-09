"""Portfolio optimizer for BBOB f1 (Sphere) on [-5,5]^dim.

Strategy:
  1. A small deterministic global net (LHS + random) to seed local solvers.
  2. CMA-ES from the best global point with a generous budget (adapts to
     rotation/shift via covariance adaptation).
  3. L-BFGS-B polish from several diverse starts.
  4. A final CMA-ES squeeze with the remaining budget.

The best point found is always returned.  A hard wall-clock deadline keeps
the process well inside the 330 s kill limit.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start (LHS + random) with CMA-ES / L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin so the process is never killed at 300/330s.
        deadline = t0 + 280.0

        def time_ok():
            return time.time() < deadline

        def wrap(z):
            x = np.clip(np.asarray(z, dtype=float).reshape(-1), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return 1e30

        rng = np.random.default_rng(0)

        # Deterministic Latin-hypercube spread (global coverage).
        # Reduced to start CMA-ES earlier for better early-progress AOCC.
        n_lhs = 20
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        # A handful of extra random starts.
        n_rand = 10
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        # Include origin (often near the shifted minimum for Sphere).
        origin = np.zeros(dim)

        candidates = np.vstack([lhs, rand, origin.reshape(1, -1)])

        # Order candidates so "central / easy" points are tried first.
        scores = np.linalg.norm(candidates, axis=1)
        order = np.argsort(scores)
        candidates = candidates[order]

        best_x = candidates[0]
        best_f = wrap(best_x)
        budget_left = budget - 1

        # ------------------------------------------------------------------
        # Phase A: CMA-ES from the best of the global net (global basin finder).
        # ------------------------------------------------------------------
        try:
            from cma import CMA
            if budget_left > 60:
                cma_budget = int(min(300, max(100, budget_left * 0.65)))
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=1.5,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=cma_budget,
                          seed=1,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-14,
                              'tolx'    : 0.0,
                              'tolopt'  : 0.0,
                          })
                budget_left -= int(res.evaluations)
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase B: fast L-BFGS-B polish from a spread of starts.
        # ------------------------------------------------------------------
        try:
            from scipy.optimize import minimize
            polish_starts = [best_x]
            # Add diverse global candidates.
            for idx in [0, len(candidates) // 4, len(candidates) // 2,
                        3 * len(candidates) // 4, len(candidates) - 1]:
                if idx < len(candidates):
                    polish_starts.append(candidates[idx])
            # De-duplicate close starts.
            kept = []
            for s in polish_starts:
                if all(np.linalg.norm(s - k) > 1e-6 for k in kept):
                    kept.append(s)
            polish_starts = kept

            for s in polish_starts:
                if budget_left <= 5 or not time_ok():
                    break
                try:
                    r = minimize(wrap, s, method='L-BFGS-B',
                                 bounds=[(-5.0, 5.0)] * dim,
                                 options={'maxiter': 500,
                                          'maxfun': max(5, min(250, budget_left))})
                    budget_left -= int(r.nfev)
                    if r.fun <= best_f:
                        best_f = float(r.fun)
                        best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase C: final squeeze with whatever budget remains (CMA-ES again).
        # ------------------------------------------------------------------
        if budget_left > 30 and time_ok():
            try:
                from cma import CMA
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=0.3,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=max(20, budget_left),
                          seed=2,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-14,
                              'tolx'    : 0.0,
                              'tolopt'  : 0.0,
                          })
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase D: High-precision Newton-like polish using centered quadratic
        # model fit. Sphere is exactly quadratic, so a local quadratic fit
        # gives super-linear convergence to the minimum.
        # ------------------------------------------------------------------
        if budget_left > 10 and time_ok():
            try:
                # Sample a small local neighborhood around best_x to fit
                # a centered quadratic model.
                h = 0.01  # small step for local quadratic approximation
                n_pts = 2 * dim + 1
                pts = []
                vals = []
                # Center point
                pts.append(best_x.copy())
                vals.append(wrap(best_x))
                budget_left -= 1
                # Forward differences along each coordinate
                for i in range(dim):
                    for sign in [1, -1]:
                        p = best_x.copy()
                        p[i] += sign * h
                        p = np.clip(p, -5.0, 5.0)
                        pts.append(p)
                        vals.append(wrap(p))
                        budget_left -= 1

                pts = np.array(pts)
                vals = np.array(vals)

                # Center the points around best_x for numerical stability
                centered = pts - best_x
                # Build design matrix for quadratic model:
                # f(x) ≈ c0 + c^T * d + 0.5 * d^T * H * d
                # where d = x - best_x
                n = len(pts)
                # Design matrix columns: [1, d_1, ..., d_d, d_1^2, d_1*d_2, ..., d_d^2]
                d_cols = centered  # n x dim
                quad_cols = np.zeros((n, dim * (dim + 1) // 2))
                idx = 0
                for i in range(dim):
                    for j in range(i, dim):
                        if i == j:
                            quad_cols[:, idx] = centered[:, i] ** 2
                        else:
                            quad_cols[:, idx] = 2.0 * centered[:, i] * centered[:, j]
                        idx += 1

                A = np.hstack([np.ones((n, 1)), d_cols, quad_cols])
                # Least squares fit
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
                    c0 = coeffs[0]
                    c = coeffs[1:1 + dim]
                    H_flat = coeffs[1 + dim:]
                    # Reconstruct Hessian
                    H = np.zeros((dim, dim))
                    idx = 0
                    for i in range(dim):
                        for j in range(i, dim):
                            if i == j:
                                H[i, i] = H_flat[idx]
                            else:
                                H[i, j] = H_flat[idx]
                                H[j, i] = H_flat[idx]
                            idx += 1

                    # Newton step: solve H * delta = -c
                    try:
                        delta = np.linalg.solve(H + 1e-10 * np.eye(dim), -c)
                    except np.linalg.LinAlgError:
                        delta = np.linalg.lstsq(H + 1e-10 * np.eye(dim), -c, rcond=None)[0]

                    new_x = np.clip(best_x + delta, -5.0, 5.0)
                    new_f = wrap(new_x)
                    budget_left -= 1
                    if new_f <= best_f:
                        best_f = float(new_f)
                        best_x = new_x
                except Exception:
                    pass
            except Exception:
                pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
