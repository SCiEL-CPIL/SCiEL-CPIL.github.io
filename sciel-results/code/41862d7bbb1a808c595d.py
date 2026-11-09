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
            if time.time() > deadline:
                return 1e30
            x = np.clip(np.asarray(z, dtype=float).reshape(-1), -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return 1e30

        rng = np.random.default_rng(0)

        # Deterministic Latin-hypercube spread (global coverage).
        n_lhs = 24
        lhs = np.empty((n_lhs, dim))
        for j in range(dim):
            order = rng.permutation(n_lhs)
            lhs[:, j] = -5.0 + 10.0 * (order + 0.5) / n_lhs

        # A handful of extra random starts.
        n_rand = 7
        rand = rng.uniform(-5.0, 5.0, size=(n_rand, dim))

        # Explicitly include the origin (Sphere minima after shift often near center).
        origin = np.zeros(dim)

        candidates = np.vstack([lhs, rand, [origin]])

        # Order the deterministic candidates so the "central / easy" points are
        # tried first (cheap global heuristic that often pays early).
        scores = np.linalg.norm(candidates + 1.0, axis=1)
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
                # Give CMA-ES a generous budget; it adapts to rotation/shift.
                cma_budget = int(min(350, max(80, budget_left * 0.70)))
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=1.5,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=cma_budget,
                          seed=1,
                          popsize=10,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-12,
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
            # Polishing starts: current best + a spread of global candidates.
            polish_starts = [best_x]
            polish_starts += [candidates[0], candidates[len(candidates) // 4],
                              candidates[3 * len(candidates) // 4],
                              candidates[-1]]
            # De-duplicate close starts to spend budget usefully.
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
                                 options={'maxiter': 400,
                                          'maxfun': max(5, min(200, budget_left))})
                    budget_left -= int(r.nfev)
                    if r.fun <= best_f:
                        best_f = float(r.fun)
                        best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
                except Exception:
                    pass
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Phase B2: Nelder-Mead polish (fewer calls/iteration, good for
        #        bridging CMA-ES result to L-BFGS-B precision).
        # ------------------------------------------------------------------
        if budget_left > 15 and time_ok():
            try:
                from scipy.optimize import minimize
                nm_budget = max(10, min(80, budget_left // 2))
                r = minimize(wrap, best_x, method='Nelder-Mead',
                             options={'maxiter': 200,
                                      'maxfun': nm_budget,
                                      'xatol': 1e-10,
                                      'fatol': 1e-12,
                                      'adaptive': True})
                budget_left -= int(r.nfev)
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase C: final high-precision polish (L-BFGS-B with tight tolerance).
        # ------------------------------------------------------------------
        if budget_left > 10 and time_ok():
            try:
                from scipy.optimize import minimize
                r = minimize(wrap, best_x, method='L-BFGS-B',
                             bounds=[(-5.0, 5.0)] * dim,
                             options={'maxiter': 1000,
                                      'maxfun': max(10, budget_left),
                                      'gtol': 1e-14,
                                      'ftol': 1e-15})
                budget_left -= int(r.nfev)
                if r.fun <= best_f:
                    best_f = float(r.fun)
                    best_x = np.clip(np.asarray(r.x, dtype=float).reshape(-1), -5.0, 5.0)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Phase D: optional final CMA-ES squeeze if budget remains.
        # ------------------------------------------------------------------
        if budget_left > 20 and time_ok():
            try:
                from cma import CMA
                res = CMA(evolve=wrap, x0=best_x.tolist(),
                          sigma0=0.05,
                          bounds=[(-5.0,) * dim, (5.0,) * dim],
                          maxfevals=max(10, budget_left),
                          seed=3,
                          verbose=-9, quiet=True,
                          pickle=False,
                          toff=0, cmaes_tolerances={
                              'tolask'  : 0.0,
                              'tolfun'  : 1e-15,
                              'tolx'    : 0.0,
                              'tolopt'  : 0.0,
                          })
                if res.fbest <= best_f:
                    best_f = float(res.fbest)
                    best_x = np.array(res.xbest, dtype=float).reshape(dim)
            except Exception:
                pass

        return best_x, float(best_f)
# EVOLVE-BLOCK-END
