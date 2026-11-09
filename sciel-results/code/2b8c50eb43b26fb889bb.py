"""Multi-objective optimizer for RE21: HV-driven evolutionary search + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven multi-objective search with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        def dominated_mask(Fv):
            """mask[i] = True if Fv[i] is dominated by some other row."""
            n = len(Fv)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                d = (Fv <= Fv[i]) & (Fv < Fv[i])
                if d.sum() > 1:  # at least one other row dominates i
                    dom[i] = True
            return dom

        # 1) Latin Hypercube initialization for diversity
        n_init = min(40, budget // 3)
        rng = np.random.default_rng(0)

        def lhs_sample(n, d):
            """Latin Hypercube Sampling."""
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points

        X_init = lo + span * lhs_sample(n_init, dim)
        for x in X_init:
            X.append(x.copy())
            F.append(eval_pt(x))

        if used >= budget:
            return np.array(X)

        # 2) Evolutionary loop: HV-contribution driven parent selection
        while used < budget:
            if time.time() > deadline:
                break
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Compute HV contribution (2D) for each non-dominated point
            nd = Fv[nd_idx]
            n_nd = len(nd)
            hv_contrib = np.zeros(n_nd)
            # Sort by first objective ascending
            order = np.argsort(nd[:, 0])
            nd_sorted = nd[order]
            # For 2D: contribution of point i = (ref0 - f0_i) * (f1_prev - f1_i)
            # where f1_prev is the f1 of the point with the next-larger f0 (or ref1)
            ref = np.array([3000.0, 0.06])
            prev_f1 = ref[1]
            for k in range(n_nd - 1, -1, -1):
                f0, f1 = nd_sorted[k]
                # Only count if strictly better than ref
                if f0 < ref[0] and f1 < ref[1]:
                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                    hv_contrib[order[k]] = max(contrib, 0.0)
                else:
                    hv_contrib[order[k]] = 0.0
                prev_f1 = f1

            total_hv = hv_contrib.sum()
            if total_hv <= 0:
                # Fallback: pick random non-dominated
                p = rng.integers(len(nd_idx))
            else:
                probs = hv_contrib / total_hv
                p = rng.choice(len(nd_idx), p=probs)

            parent = Xv[nd_idx[p]]

            # Adaptive mutation step
            gen = used - n_init
            total_gens = max(1, budget - n_init)
            step = max(0.02, 0.12 * (1.0 - gen / total_gens))
            child = np.clip(parent + step * span * rng.standard_normal(dim), lo, hi)
            fc = eval_pt(child)

            # Replace the point with the lowest HV contribution
            worst = int(np.argmin(hv_contrib))
            X[nd_idx[worst]] = child
            F[nd_idx[worst]] = fc

        # 3) Local polish: L-BFGS-B on non-dominated points with weighted sums
        if time.time() < deadline - 10.0 and used < budget:
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Weight vectors for weighted-sum (2 objectives)
            n_w = min(8, budget - used)
            if n_w > 0:
                weights = np.array([[1.0 - i / max(n_w - 1, 1), i / max(n_w - 1, 1)]
                                    for i in range(n_w)])

                for w in weights:
                    if used >= budget or time.time() > deadline:
                        break

                    def obj_func(x, w=w):
                        f = np.asarray(problem(x), dtype=float)
                        return float(np.dot(w, f))

                    # Start from the best point for this weight
                    Fv2 = np.array(F)
                    scores = Fv2[nd_idx] @ w
                    best_local = nd_idx[int(np.argmin(scores))]
                    x0 = Xv[best_local].copy()

                    res = minimize(obj_func, x0, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 50, 'gtol': 1e-6})
                    if res.success and used < budget:
                        xf = np.clip(res.x, lo, hi)
                        ff = eval_pt(xf)
                        # Replace worst HV point
                        Fv3 = np.array(F)
                        dom3 = dominated_mask(Fv3)
                        nd3 = np.where(~dom3)[0]
                        if len(nd3) > 0:
                            nd3_f = Fv3[nd3]
                            ref = np.array([3000.0, 0.06])
                            # Compute HV contributions to find worst
                            n_nd3 = len(nd3)
                            hv_contrib3 = np.zeros(n_nd3)
                            order3 = np.argsort(nd3_f[:, 0])
                            nd3_sorted = nd3_f[order3]
                            prev_f1 = ref[1]
                            for k in range(n_nd3 - 1, -1, -1):
                                f0, f1 = nd3_sorted[k]
                                if f0 < ref[0] and f1 < ref[1]:
                                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                                    hv_contrib3[order3[k]] = max(contrib, 0.0)
                                else:
                                    hv_contrib3[order3[k]] = 0.0
                                prev_f1 = f1

                            worst3 = int(np.argmin(hv_contrib3))
                            X[nd3[worst3]] = xf
                            F[nd3[worst3]] = ff

        return np.array(X)
# EVOLVE-BLOCK-END
