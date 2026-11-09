"""Portfolio optimizer for BBOB f1 (Sphere) on [-5,5]^dim.

Global net of candidate starts (deterministic Latin-hypercube + randoms),
each polished by a fast local solver (CMA-ES, L-BFGS-B) under a hard
wall-clock deadline.  The best point found so far is always returned.
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
        n_lhs = 30
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        # Extra random starts.
        n_rand = 15
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        # Add a few structured points: center, axis-aligned midpoints.
        center = np.zeros(dim)
        axis_pts = []
        for d in range(dim):
            for sign in [-1.0, 1.0]:
                p = np.zeros(dim)
                p[d] = sign * 2.5
                axis_pts.append(p)
        structured = np.array([center] + axis_pts)

        candidates = np.vstack([lhs, rand, structured])

        # Evaluate all candidates to find the best starting point early.
        # This gives us a good global picture before local optimization.
        eval_budget = min(len(candidates), max(30, budget // 4))
        # Order: try center first, then LHS, then randoms, then structured
        # Actually, let's just evaluate a subset deterministically ordered by
        # distance from origin (center tends to be good for shifted sphere).
        dists = np.linalg.norm(candidates, axis=1)
        order = np.argsort(dists)
        candidates_ordered = candidates[order]

        best_x = candidates_ordered[0]
        best_f = wrap(best_x)
        budget_left = budget - 1

        # Evaluate more candidates to find a better global start
        for idx in range(1, min(eval_budget, len(candidates_ordered))):
            if budget_left <= 10 or not time_ok():
                break
            c = candidates_ordered[idx]
            f_val = wrap(c)
            budget_left -= 1
            if f_val < best_f:
                best_f = f_val
                best_x = c.copy()

        # ------------------------------------------------------------------
        # Phase A: CMA-ES from the best of the global net (global basin finder).
        # ------------------------------------------------------------------
        try:
            from cma import CMA
            if budget_left > 80:
                cma_budget = int(min(150, max(30, budget_left * 0.30)))
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
            # Polishing starts: current best + several diverse global candidates.
            polish_starts = [best_x]
            # Add diverse points from the ordered candidate list
            n_cand = len(candidates_ordered)
            for frac in [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]:
                idx = int(frac * (n_cand - 1))
                polish_starts.append(candidates_ordered[idx])
            # De-duplicate close starts to spend budget usefully.
            kept = []
            for s in polish_starts:
                s = np.asarray(s, dtype=float).reshape(-1)
                if all(np.linalg.norm(s - k) > 0.01 for k in kept):
                    kept.append(s)
            polish_starts = kept

            for s in polish_starts:
                if budget_left <= 10 or not time_ok():
                    break
                try:
                    maxfun = max(10, min(300, budget_left))
                    r = minimize(wrap, s, method='L-BFGS-B',
                                 bounds=[(-5.0, 5.0)] * dim,
                                 options={'maxiter': 500,
                                          'maxfun': maxfun,
                                          'ftol': 1e-12})
                    budget_left -= int(r.nfev)
                    if r.fun <= best_f:
                        best_f = float(r.fun)
                        best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase C: final high-precision polish.
        # ------------------------------------------------------------------
        if budget_left > 20 and time_ok():
            try:
                from scipy.optimize import minimize
                # Nelder-Mead for final local refinement (derivative-free,
                # can squeeze out last bits of precision).
                r = minimize(wrap, best_x, method='Nelder-Mead',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 1000,
                                      'maxfev': max(20, min(200, budget_left)),
                                      'xatol': 1e-12, 'fatol': 1e-12,
                                      'adaptive': True})
                budget_left -= int(r.nfev)
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        # Final L-BFGS-B polish with very tight tolerance if budget remains.
        if budget_left > 10 and time_ok():
            try:
                from scipy.optimize import minimize
                r = minimize(wrap, best_x, method='L-BFGS-B',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 1000,
                                      'maxfun': max(5, min(100, budget_left)),
                                      'ftol': 1e-15,
                                      'gtol': 1e-12})
                budget_left -= int(r.nfev)
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
