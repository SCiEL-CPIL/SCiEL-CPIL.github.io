"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted multi-objective Bayesian optimizer (EHVI-guided)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        def time_left():
            return deadline - time.time()

        def valid(x):
            return np.all(np.isfinite(x)) and np.all(x >= lo) and np.all(x <= hi)

        def in_bounds(x):
            return np.all(x >= lo) and np.all(x <= hi)

        # ---- Hypervolume helpers (exact, 3 objectives) ----
        def hv_3d(A):
            # A: (n,3) of positive contributions; returns hypervolume
            if len(A) == 0:
                return 0.0
            A = np.asarray(A)
            order = np.lexsort((A[:, 1], A[:, 0]))
            A = A[order]
            best2 = np.full(len(A), -np.inf)
            vol = 0.0
            for i in range(len(A)):
                a0, a1, a2 = A[i]
                # rectangle in z2 over z1 in [0,a1) at height a2, plus strip in z1
                # contribution from z1 in [0, a1]: height = min(a2, best2_so_far?) use incremental
                # Use: area under curve of running max of a2 over z1
                vol += a2 * a1  # provisional, corrected below
            # Recompute correctly via running max
            vol = 0.0
            running = 0.0
            for i in range(len(A)):
                a0, a1, a2 = A[i]
                if a2 > running:
                    # width in z1 from previous z1 to current z1 at old running height
                    # we integrate over z1: piecewise constant running max
                    pass
            # Proper sweep: sort by z0, keep running max of z2, integrate z1*z2
            # Simpler exact 3D: for each point, contribution = a1*a2 minus overlaps is hard.
            # Use standard O(n^2) incremental:
            vol = 0.0
            prev_z1 = 0.0
            running_max_z2 = 0.0
            for i in range(len(A)):
                a0, a1, a2 = A[i]
                if a2 > running_max_z2:
                    # the region z1 in [prev_z1, a1) was at height running_max_z2
                    # but we need to account z1 dimension properly. Use 2D area in (z1,z2)
                    # integrated over z0 is not separable. Use the known exact 3D algorithm:
                    pass
            # Use the exact algorithm via "max over z2 for each (z0,z1)" grid is not exact.
            # Fall back to a correct O(n^2) incremental 3D hypervolume:
            vol = 0.0
            # sort by z0
            for i in range(len(A)):
                a0, a1, a2 = A[i]
                # contribution of point i's box not covered by any later (larger z0) point
                # covered if exists j with A[j,0] >= a0? We sorted ascending z0, so later j have z0>=a0
                covered_z1z2 = 0.0
                for j in range(i + 1, len(A)):
                    b0, b1, b2 = A[j]
                    # overlap in z1,z2 between box i (z1 in [0,a1], z2 in [0,a2]) and box j
                    ov1 = min(a1, b1)
                    ov2 = min(a2, b2)
                    if ov1 > 0 and ov2 > 0:
                        covered_z1z2 += ov1 * ov2
                        # subtract double-count overlaps between j and k handled by inclusion-exclusion
                        # For exactness we need full inclusion-exclusion; do it:
                # This is getting complex; use a robust exact method:
                pass
            return self._hv_exact(A)

        def _hv_exact(self, A):
            # Exact 3-objective hypervolume via dominance counting (O(n^2))
            # Contribution of point i = a1*a2 - (area in z1z2 dominated by any point with z0>=a0)
            n = len(A)
            vol = 0.0
            for i in range(n):
                a0, a1, a2 = A[i]
                # area of [0,a1]x[0,a2] covered by union of boxes j (z0_j >= a0)
                # covered if point j dominates i in z1,z2 partially
                # union area of rectangles [0,b1]x[0,b2] for j with b0>=a0
                # compute union area of axis-aligned rectangles from origin
                rects = []
                for j in range(n):
                    b0, b1, b2 = A[j]
                    if b0 >= a0:
                        rects.append((b1, b2))
                # union area of rectangles [0,w]x[0,h]
                # = area of (w_max, h_max) union -> equals max over? Not simply.
                # Union of origin-anchored rectangles: area = sum over "corners"
                # The union of origin-anchored rectangles equals the region under the
                # upper envelope; area = integral of min over ... Actually union of
                # [0,w_k]x[0,h_k] has area = sum_{k} w_k*h_k - overlaps.
                # Since all anchored at origin, union area = area of the set
                # { (x,y): exists k with x<=w_k and y<=h_k }
                # = sum over sorted by w of (w_k - w_{k-1}) * max_{j>=k} h_j  (with w sorted asc)
                if rects:
                    rs = sorted(rects, key=lambda r: r[0])
                    area = 0.0
                    prev_w = 0.0
                    # max h from this index onward
                    maxh = 0.0
                    # process from largest w to smallest to get suffix max h
                    ws = [r[0] for r in rs]
                    hs = [r[1] for r in rs]
                    # build suffix max of h
                    suff = [0.0] * len(rs)
                    m = 0.0
                    for k in range(len(rs) - 1, -1, -1):
                        if hs[k] > m:
                            m = hs[k]
                        suff[k] = m
                    for k in range(len(rs)):
                        area += (ws[k] - prev_w) * suff[k]
                        prev_w = ws[k]
                    vol += a1 * a2 - area
                else:
                    vol += a1 * a2
            return max(0.0, vol)

        def contribs(F):
            # returns array of (n,3) positive contributions, only points better than ref in all obj
            C = []
            for f in F:
                c = ref - f
                if np.all(c > 0):
                    C.append(c)
            if not C:
                return np.zeros((0, 3))
            return np.asarray(C)

        def hv(F):
            return _hv_exact(contribs(F))

        # ---- Surrogate: per-objective quadratic (degree-2) polynomial regression ----
        def fit_poly(X, y, degree=2):
            # Build feature matrix with all monomials up to `degree`
            n, d = X.shape
            feats = [np.ones(n)]
            for i in range(d):
                feats.append(X[:, i])
            if degree >= 2:
                for i in range(d):
                    for j in range(i, d):
                        feats.append(X[:, i] * X[:, j])
            M = np.column_stack(feats)
            # ridge solve
            lam = 1e-6 * np.eye(M.shape[1])
            try:
                coef, *_ = np.linalg.lstsq(M + 1e-12 * np.eye(M.shape[1]), y, rcond=None)
            except Exception:
                coef = np.linalg.solve(M.T @ M + lam, M.T @ y)
            return M, coef

        def predict(M, coef, Xnew):
            n, d = Xnew.shape
            feats = [np.ones(n)]
            for i in range(d):
                feats.append(Xnew[:, i])
            for i in range(d):
                for j in range(i, d):
                    feats.append(Xnew[:, i] * Xnew[:, j])
            Mn = np.column_stack(feats)
            return Mn @ coef

        # ---- EHVI acquisition via Monte-Carlo sampling of the surrogate ----
        def ehvi_candidates(Xtr, Ftr, n_cand=400, n_samp=60):
            # Fit per-objective surrogates
            models = []
            for o in range(n_obj):
                M, coef = fit_poly(Xtr, Ftr[:, o], degree=2)
                models.append((M, coef))
            # Sample candidate points (LHS-like + jitter around current best)
            rng_local = np.random.RandomState(0)
            cand = lo + (hi - lo) * rng_local.random((n_cand, dim))
            # Add points near current non-dominated solutions
            nd_idx = non_dominated_idx(Ftr)
            if len(nd_idx) > 0:
                for _ in range(min(100, n_cand // 4)):
                    base = Xtr[nd_idx[rng_local.randint(len(nd_idx))]]
                    p = base + 0.05 * (hi - lo) * rng_local.standard_normal(dim)
                    cand = np.vstack([cand, p])
            cand = cand[:n_cand]
            cand = np.clip(cand, lo, hi)
            # Predict mean
            mean = np.column_stack([predict(M, c, cand) for (M, c) in models])
            # Estimate uncertainty: residual std per objective
            unc = np.zeros(n_obj)
            for o in range(n_obj):
                M, c = models[o]
                resid = Ftr[:, o] - predict(M, c, Xtr)
                unc[o] = np.std(resid) + 1e-9
            # Sample surrogate outputs
            best_ehvi = -np.inf
            best_cand = cand[0]
            base_hv = hv(Ftr)
            for s in range(n_samp):
                samp = mean + unc * np.random.standard_normal((n_cand, n_obj))
                # For each candidate, EHVI = E_hv(Ftr U {samp}) - base_hv
                # Approximate by direct hv of augmented set (cheap for small n)
                for k in range(n_cand):
                    aug = np.vstack([Ftr, samp[k:k + 1]])
                    h = hv(aug)
                    eh = h - base_hv
                    if eh > best_ehvi:
                        best_ehvi = eh
                        best_cand = cand[k]
            return best_cand

        def non_dominated_idx(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # ---- Local polish: Nelder-Mead on scalarized objective (augmented Tchebycheff) ----
        def polish(x0, w):
            def obj(x):
                if not in_bounds(x):
                    return 1e6
                f = problem(x)
                return np.max(w * f)
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
                if in_bounds(res.x) and np.isfinite(res.fun):
                    return res.x
            except Exception:
                pass
            return x0

        # ---- Main loop ----
        rng = np.random.RandomState(0)
        X = []
        F = []
        used = 0

        def add(x, f):
            nonlocal used
            X.append(x)
            F.append(f)
            used += 1

        # Initial random samples
        n_init = min(24, budget)
        X0 = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X0:
            f = problem(x)
            add(x, f)
            if used >= budget or time_left() < 20:
                break

        F = np.asarray(F)
        X = np.asarray(X)

        # Surrogate-assisted search loop
        while used < budget and time_left() > 30:
            # Pick next point via EHVI
            try:
                nxt = ehvi_candidates(X, F, n_cand=300, n_samp=50)
            except Exception:
                nd = non_dominated_idx(F)
                base = X[nd[rng.randint(len(nd))]] if len(nd) else X[0]
                nxt = np.clip(base + 0.1 * (hi - lo) * rng.standard_normal(dim), lo, hi)
            f = problem(nxt)
            add(nxt, f)
            X = np.vstack([X, nxt])
            F = np.vstack([F, f])

        # Final polish: scalarized local optimization from best non-dominated points
        if time_left() > 40 and used < budget:
            nd = non_dominated_idx(F)
            # A few weight vectors to spread the front
            weights = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0]) / 3.0,
                np.array([1.0, 2.0, 1.0]) / 4.0,
                np.array([1.0, 1.0, 2.0]) / 4.0,
            ]
            for w in weights:
                if used >= budget or time_left() < 15:
                    break
                # start from the point that best matches this weight (min w·f)
                scores = F @ w
                i0 = int(np.argmin(scores))
                x0 = X[i0]
                xp = polish(x0, w)
                if used < budget and time_left() > 10:
                    fp = problem(xp)
                    add(xp, fp)
                    X = np.vstack([X, xp])
                    F = np.vstack([F, fp])

        return X
# EVOLVE-BLOCK-END
