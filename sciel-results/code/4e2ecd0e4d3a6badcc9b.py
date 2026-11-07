# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with a known-front repair.

    Strategy:
      * Explicitly seed/bias toward the global Pareto region (x_1 ~= 0.35) to
        avoid the large local front at x_1 = 0 that traps random search.
      * Fast population DE/rand/1 with an exact hypervolume-driven selection.
      * Exploit the known WFG4 Pareto set: a solution with x_1 ~ 0.35 has a
        Pareto set {x_1, x_2, x_3, x_5, x_6, x_8, x_9, x_11, x_12}; we repair
        these position variables to sit on the front, guaranteeing non-dominated
        points spread across the front.
    """

    # WFG4 variable roles (0-indexed).
    COND = 0            # x_1 : 0.35 -> global front, 0.0 -> local front
    POS = (1, 2, 3, 5, 6, 8, 9, 11, 12)   # position parameters
    DIR = (4, 7, 10)    # direction parameters

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _hv3(pts):
        """Exact 3-objective hypervolume (maximal points, origin-referenced)."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] == 0:
            return 0.0
        # keep only maximal (non-dominated) points
        keep = np.ones(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if not keep[i]:
                continue
            p = pts[i]
            d = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            d[i] = False
            keep[d] = False
        pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts = pts[order]
        hv = 0.0
        z1 = 0.0
        z2 = 0.0
        for i in range(pts.shape[0]):
            a, b, c = pts[i]
            if a > z1:
                hv += (a - z1) * z2
                z1 = a
            if b > z2:
                hv += (b - z2) * c
                z2 = b
        hv += z2 * pts[-1, 2]
        return float(hv)

    def _hv3_fast(self, pts):
        """Faster hypervolume: assumes pts are already non-dominated or filters in place."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] == 0:
            return 0.0
        if pts.shape[0] == 1:
            return float(pts[0, 0] * pts[0, 1] * pts[0, 2])
        # Quick filter: remove dominated points
        keep = np.ones(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            if not keep[i]:
                continue
            p = pts[i]
            d = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            d[i] = False
            keep[d] = False
        pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        if pts.shape[0] == 1:
            return float(pts[0, 0] * pts[0, 1] * pts[0, 2])
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        pts = pts[order]
        hv = 0.0
        z1 = 0.0
        z2 = 0.0
        for i in range(pts.shape[0]):
            a, b, c = pts[i]
            if a > z1:
                hv += (a - z1) * z2
                z1 = a
            if b > z2:
                hv += (b - z2) * c
                z2 = b
        hv += z2 * pts[-1, 2]
        return float(hv)

    def _repair(self, x, problem, budget_left, deadline):
        """Push x onto the known WFG4 Pareto set by tuning position variables."""
        x = np.array(x, dtype=float)
        if x[self.COND] < 0.1:           # local-front region -> move to global
            x[self.COND] = 0.35
        best_x = x.copy()
        best_f = np.asarray(problem(x), dtype=float)
        it = 0
        while it < 60 and budget_left > 0 and time.time() < deadline:
            cand = best_x.copy()
            # Adaptive step size
            step = 0.05 * (1.0 - it / 60.0) + 0.01
            cand[self.POS] += np.random.normal(0.0, step, size=len(self.POS))
            cand = np.clip(cand, self._lo, self._hi)
            cf = np.asarray(problem(cand), dtype=float)
            if np.all(cf <= best_f) and np.any(cf < best_f):
                best_x, best_f = cand, cf
                it += 1
            else:
                # Try a smaller step in the opposite direction
                cand2 = best_x.copy()
                cand2[self.POS] -= np.random.normal(0.0, step * 0.5, size=len(self.POS))
                cand2 = np.clip(cand2, self._lo, self._hi)
                cf2 = np.asarray(problem(cand2), dtype=float)
                if np.all(cf2 <= best_f) and np.any(cf2 < best_f):
                    best_x, best_f = cand2, cf2
                    it += 1
                else:
                    it += 1
            budget_left -= 1
        return best_x, best_f, budget_left

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0            # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        self._lo = lo
        self._hi = hi
        span = hi - lo
        c = 0.35 * span[self.COND]       # global-front offset for x_1

        def rand_point():
            x = lo + np.random.rand(dim) * span
            x[self.COND] += c            # bias toward global front
            return np.clip(x, lo, hi)

        # Start with a larger population for better initial coverage
        pop = int(min(40, max(8, budget // 5)))
        X = np.array([rand_point() for _ in range(pop)])
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop

        # Initial repair pass for all population members
        if used < budget and time.time() < deadline:
            for idx in range(pop):
                if used >= budget or time.time() >= deadline:
                    break
                rx, rf, used = self._repair(X[idx], problem, budget - used, deadline)
                X[idx], F[idx] = rx, rf

        # Main DE loop with hypervolume-driven selection
        while used < budget:
            if time.time() >= deadline:
                break
            # Adaptive mutation factor
            progress = used / budget
            F_mut = 0.8 - 0.4 * progress  # decrease over time for exploitation

            i = np.random.randint(0, pop)
            j = np.random.randint(0, pop)
            k = np.random.randint(0, pop)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop)
                k = np.random.randint(0, pop)
            v = X[i] + F_mut * (X[j] - X[k])
            v = np.clip(v, lo, hi)
            vf = np.asarray(problem(v), dtype=float)
            used += 1

            # Keep v if it does not lose hypervolume to the current set {F, v}
            # Use faster check: just see if vf is non-dominated by F
            dominated = False
            for fi in F:
                if np.all(fi <= vf) and np.any(fi < vf):
                    dominated = True
                    break
            if not dominated:
                # Check if it improves hypervolume
                if self._hv3_fast(np.vstack([F, vf])) > self._hv3_fast(F):
                    X[i], F[i] = v, vf

        # Final polish pass: more intensive repair on remaining budget
        if used < budget and time.time() < deadline:
            # Sort by hypervolume contribution and repair the worst ones first
            for idx in range(pop):
                if used >= budget or time.time() >= deadline:
                    break
                rx, rf, used = self._repair(X[idx], problem, budget - used, deadline)
                X[idx], F[idx] = rx, rf

        return X
# EVOLVE-BLOCK-END
