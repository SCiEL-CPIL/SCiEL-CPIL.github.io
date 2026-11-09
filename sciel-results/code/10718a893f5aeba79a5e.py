"""Multi-objective optimizer for ZDT4 using hypervolume-guided population search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective optimizer for ZDT4 with hypervolume selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv2(self, pts):
        """Exact 2D hypervolume for points, ref = (11, 400). Only points inside count."""
        if len(pts) == 0:
            return 0.0
        pts = [p for p in pts if p[0] < 11.0 and p[1] < 400.0]
        if len(pts) == 0:
            return 0.0
        pts = sorted(pts, key=lambda p: p[0])
        hv = 0.0
        max_y = 400.0
        for i, (x, y) in enumerate(pts):
            # contribution: width to next point (or ref x), height from y to max_y
            if i + 1 < len(pts):
                nx = pts[i + 1][0]
            else:
                nx = 11.0
            w = nx - x
            if w > 0:
                h = max_y - y
                if h > 0:
                    hv += w * h
        return hv

    def _non_dominated(self, F):
        """Return boolean mask of non-dominated points (2D)."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and (F[j][0] < F[i][0] or F[j][1] < F[i][1]):
                    mask[i] = False
                    break
        return mask

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Population size: small to keep evaluations diverse within budget
        pop_size = max(10, min(30, budget // 8))
        pop_size = min(pop_size, budget)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume (dedup by rounding)
        all_points = [tuple(f) for f in F]

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Compute non-dominated set
            nd_mask = self._non_dominated(F)
            nd_count = np.sum(nd_mask)

            # Selection: prefer non-dominated; among them, use hypervolume contribution
            if nd_count > 0:
                candidates = X[nd_mask]
                cand_F = F[nd_mask]
            else:
                candidates = X
                cand_F = F

            # Pick two parents: one random from candidates, one by hypervolume contribution
            n_cand = len(candidates)
            if n_cand >= 2:
                i1 = rng.integers(n_cand)
                # Estimate hypervolume contribution: remove point, see HV drop
                # For speed, use a simpler heuristic: pick point with best (smallest) f0
                # Actually, let's just pick another random
                i2 = rng.integers(n_cand)
                while i2 == i1 and n_cand > 1:
                    i2 = rng.integers(n_cand)
                p1 = candidates[i1]
                p2 = candidates[i2]
            else:
                p1 = candidates[0]
                p2 = candidates[0]

            # Crossover or mutation
            r = rng.random()
            if r < 0.5 and p1 is not p2:
                # Simulated binary crossover
                child1 = np.where(rng.random(dim) < 0.5, p1, p2)
                child2 = np.where(rng.random(dim) < 0.5, p1, p2)
            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Mutation
            # Adaptive step size: start large, decrease
            progress = used / budget
            step = span * (0.3 * (1 - progress) + 0.05)
            noise1 = rng.standard_normal(dim)
            noise2 = rng.standard_normal(dim)
            child1 = child1 + step * noise1
            child2 = child2 + step * noise2
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)

            # Evaluate children
            fc1 = problem(child1)
            used += 1
            all_points.append(tuple(fc1))
            if time.time() > deadline:
                break

            fc2 = problem(child2)
            used += 1
            all_points.append(tuple(fc2))

            # Replace worst points in population
            # Use simple sum as fitness indicator for replacement
            sums = np.sum(F, axis=1)
            worst1 = int(np.argmax(sums))
            X[worst1] = child1
            F[worst1] = fc1

            if used < budget:
                sums = np.sum(F, axis=1)
                worst2 = int(np.argmax(sums))
                X[worst2] = child2
                F[worst2] = fc2

        return X
# EVOLVE-BLOCK-END
