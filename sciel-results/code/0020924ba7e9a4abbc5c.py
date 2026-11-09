"""Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with adaptive perturbation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0          # evaluator kills at 330s; keep a 30s margin
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume (larger volume == better point).
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---------- helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization). sub: (n,2), ref2: (2,)"""
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                hi0 = min(x, ref2[0])
                w0 = hi0 - prev
                if w0 <= 0:
                    continue
                # best (minimum) y among points with obj0 >= x
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                if ybest >= ref2[1]:
                    prev = x
                    continue
                total += w0 * (ref2[1] - ybest)
                prev = x
            return total

        def hv3(P, ref3):
            """3D hypervolume (minimization). P: (n,3), ref3: (3,)"""
            if len(P) == 0:
                return 0.0
            mask = np.all(P < ref3, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                hi0 = min(x, ref3[0])
                w0 = hi0 - prev
                if w0 <= 0:
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev = x
            return total

        def pareto(F):
            """Return indices of non-dominated points."""
            n = len(F)
            keep = []
            for i in range(n):
                fi = F[i]
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        # ---------- main loop ----------
        # Start with a random spread.
        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # seed points - use a structured initial set for better coverage
        # Latin hypercube-like initialization for diversity
        for _ in range(10):
            x = lo + (hi - lo) * rng.random(dim)
            add(x, problem(x))

        used = len(F)
        # adaptive perturbation scale
        scale = 0.2

        # Track hypervolume to guide exploration
        last_hv = 0.0
        hv_stagnation = 0

        while used < budget:
            # deadline guard
            if time.time() > deadline - 5.0:
                break

            # Compute current hypervolume of all recorded points
            F_arr = np.array(F)
            current_hv = hv3(F_arr, ref)
            
            # If hypervolume improved, reset stagnation counter
            if current_hv > last_hv + 1e-6:
                last_hv = current_hv
                hv_stagnation = 0
            else:
                hv_stagnation += 1

            # Choose strategy based on stagnation
            if hv_stagnation > 50:
                # Increase exploration: larger steps, more random restarts
                if rng.random() < 0.3:
                    # Random restart
                    x = lo + (hi - lo) * rng.random(dim)
                    fc = problem(x)
                    add(x, fc)
                    used += 1
                    scale = max(0.3, scale * 1.5)
                    continue
                step_mult = 2.0
            elif hv_stagnation > 20:
                step_mult = 1.5
            else:
                step_mult = 1.0

            # choose parents: prefer non-dominated
            keep = pareto(F_arr)
            if len(keep) == 0:
                keep = list(range(len(F_arr)))

            # select parents weighted toward low objective sum among non-dominated
            cand = F_arr[keep]
            sums = np.sum(cand, axis=1)
            range_sums = sums.max() - sums.min() + 1e-9
            w = np.exp(-2.0 * (sums - sums.min()) / range_sums)
            w /= w.sum()
            pidx = keep[int(rng.choice(len(keep), p=w))]

            # mutation: adaptive step
            step = scale * step_mult
            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X[pidx] + delta
            child = np.clip(child, lo, hi)
            fc = problem(child)
            add(child, fc)
            used += 1

            # update scale: shrink slowly to refine
            scale = max(0.03, scale * 0.995)

        return X
# EVOLVE-BLOCK-END
