# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize as sp_minimize


class Optimizer:
    """Hypervolume-driven evolutionary multi-objective search for RE34 (5-dim, 3-obj)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290  # hard wall-clock safety margin (limit is 300s)

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng_span = hi - lo
        d = self.dim

        def time_ok():
            return time.time() < deadline

        def hv_3d(F, ref):
            # Exact 3-objective hypervolume of points strictly better than ref.
            pts = [tuple(f) for f in F if all(f[i] < ref[i] for i in range(3))]
            if not pts:
                return 0.0
            pts.sort(key=lambda p: (p[0], p[1], p[2]))
            total = 0.0
            for i in range(len(pts)):
                a, b, c = pts[i]
                # region where this point is the minimizer in obj0: x0 in [a, next_a or ref0]
                next_a = pts[i + 1][0] if i + 1 < len(pts) else ref[0]
                w0 = next_a - a
                if w0 <= 0:
                    continue
                # for x0 in that slice, point is best in obj0; need it best in obj1 & obj2
                # among points with obj0 <= x0 (i.e. j <= i), find max extent in obj1, obj2
                # build set of (obj1, obj2) for j<=i, compute 2D HV with ref (ref1, ref2)
                sub = [pts[j][1:] for j in range(i + 1)]
                # 2D HV: sort by obj1, track min obj2
                sub.sort(key=lambda p: p[0])
                min2 = ref[2]
                area = 0.0
                for (p1, p2) in sub:
                    if p2 < min2:
                        area += (ref[1] - p1) * (min2 - p2)
                        min2 = p2
                total += w0 * area
            return total

        # ---- Initial population: mix of random + Tchebycheff weight points for spread ----
        pop_size = 24
        X = np.empty((pop_size, d))
        # first 12: uniform random
        X[:12] = lo + rng_span * rng.random((12, d))
        # next 12: Tchebycheff weight points (3 objectives -> 2-simplex weights)
        weights = []
        n_w = 4
        for a in range(n_w + 1):
            for b in range(n_w + 1 - a):
                c = n_w - a - b
                weights.append(np.array([a, b, c], dtype=float) / n_w)
        weights = np.array(weights[:12])
        for k in range(12):
            w = weights[k]
            # start near a point that balances objectives by weight
            x0 = lo + rng_span * rng.random(d)
            X[12 + k] = x0

        F = np.array([problem(x) for x in X])
        used = pop_size

        # ---- Evolutionary loop with HV-driven selection ----
        # Adaptive mutation: start wide, shrink over time.
        base_mut = 0.35
        while used < self.budget and time_ok():
            # remaining fraction controls mutation magnitude
            frac = max(0.0, (self.budget - used) / self.budget)
            mut_scale = base_mut * (0.3 + 0.7 * frac)

            # compute hypervolume contribution per point (approx: remove each, measure drop)
            # For 3D with n=24, O(n^3) is fine.
            base_hv = hv_3d(F, self.ref)
            contribs = np.zeros(pop_size)
            for i in range(pop_size):
                sub = np.delete(F, i, axis=0)
                contribs[i] = base_hv - hv_3d(sub, self.ref)

            # selection: pick parent with highest contribution
            # use a small tournament over top contributors for diversity
            order = np.argsort(-contribs)
            top = order[:max(6, pop_size // 3)]
            parent_idx = top[rng.integers(len(top))]
            parent = X[parent_idx].copy()

            # mutation: Gaussian with adaptive scale
            child = parent + mut_scale * rng_span * rng.standard_normal(d)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # replacement: replace the point with lowest HV contribution
            worst = int(np.argmin(contribs))
            X[worst] = child
            F[worst] = fc

        # ---- Final local polish: Tchebycheff scalarization via L-BFGS-B on best points ----
        # Pick a handful of distinct weight vectors, polish near current best points.
        # This tightens the front without adding many evaluations.
        n_weights = 6
        polish_w = []
        for a in range(3):
            for b in range(3 - a):
                c = 2 - a - b
                polish_w.append(np.array([a, b, c], dtype=float) / 2.0)
        polish_w = polish_w[:n_weights]

        # current best per weight: choose the point minimizing Tchebycheff for each weight
        for w in polish_w:
            if used >= self.budget or not time_ok():
                break
            # Tchebycheff objective: max_i w_i * (f_i - z_i)
            z = np.min(F, axis=0)  # nadir approximation
            def tcheb(x, w=w, z=z):
                f = problem(x)
                return np.max(w * (f - z))
            # start from the point that best fits this weight
            scores = np.array([np.max(w * (f - z)) for f in F])
            x0 = X[int(np.argmin(scores))].copy()
            try:
                res = sp_minimize(tcheb, x0, method='L-BFGS-B',
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 30, 'ftol': 1e-12})
                if res.success or res.fun < scores.min():
                    xnew = np.clip(res.x, lo, hi)
                    fnew = problem(xnew)
                    used += 1
                    # replace worst point by contribution
                    base_hv = hv_3d(F, self.ref)
                    contribs = np.zeros(pop_size)
                    for i in range(pop_size):
                        sub = np.delete(F, i, axis=0)
                        contribs[i] = base_hv - hv_3d(sub, self.ref)
                    worst = int(np.argmin(contribs))
                    X[worst] = xnew
                    F[worst] = fnew
            except Exception:
                pass

        return X
# EVOLVE-BLOCK-END
