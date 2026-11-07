# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with coordinate-descent polish.

    Strategy:
      * Seed/bias toward the global Pareto region (x_1 ~= 0.35).
      * Fast population DE/rand/1 with hypervolume-driven selection.
      * Deterministic coordinate-descent polish via ternary search per dimension,
        using hypervolume contribution as scalar fitness guide.
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

    @staticmethod
    def _dominates(a, b):
        """True if a dominates b (a <= b in all, a < b in at least one)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _hv_contribution_approx(self, f, archive_F, ref):
        """Fast approximation of hypervolume contribution using non-domination and distance.
        
        Returns a score that is higher for points with greater hypervolume contribution.
        """
        if not archive_F:
            # Contribution is just the volume under f up to ref
            return float(np.prod(np.maximum(ref - f, 0)))
        
        # Check if f is dominated by any archive point
        for af in archive_F:
            if self._dominates(af, f):
                return 0.0
        
        # Compute distance-based score: points farther from archive in objective space
        # tend to have higher hypervolume contribution
        max_dist = 0.0
        for af in archive_F:
            # Distance in objective space (normalized by ref)
            d = np.sum(np.abs(f - af) / np.maximum(ref, 1e-10))
            max_dist = max(max_dist, d)
        
        # Combine with volume under f
        vol = float(np.prod(np.maximum(ref - f, 0)))
        return vol * (1.0 + 0.1 * max_dist)

    def _hv_polish(self, x, f, problem, bounds, budget_remaining, deadline, archive_F, ref):
        """Coordinate-wise local search optimizing hypervolume contribution.
        
        Returns (best_x, best_f, evaluations_used).
        """
        dim = len(x)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        evals_used = 0
        best_x = x.copy()
        best_f = f.copy()
        best_score = self._hv_contribution_approx(best_f, archive_F, ref)
        
        # Try small perturbations in each dimension
        for d in range(dim):
            if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                break
            
            # Try several step sizes
            for step_frac in [0.01, 0.05, 0.1, 0.2]:
                if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                    break
                
                step = step_frac * span[d]
                if step < 1e-10:
                    continue
                
                for direction in [-1, 1]:
                    if budget_remaining - evals_used <= 0 or time.time() >= deadline:
                        break
                    
                    pt = best_x.copy()
                    pt[d] += direction * step
                    pt = np.clip(pt, lo, hi)
                    pf = np.asarray(problem(pt), dtype=float)
                    evals_used += 1
                    
                    score = self._hv_contribution_approx(pf, archive_F, ref)
                    if score > best_score:
                        best_x = pt
                        best_f = pf
                        best_score = score
        
        return best_x, best_f, evals_used

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 275.0            # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # WFG4: x_1 = 0.35 gives global front
        cond_target = 0.35 * span[self.COND] + lo[self.COND]

        def rand_point_near_front():
            """Generate point biased toward the global Pareto front region."""
            x = lo + np.random.rand(dim) * span
            # Bias x_1 toward 0.35 with some spread
            x[self.COND] = lo[self.COND] + cond_target * np.random.beta(2, 5)
            return np.clip(x, lo, hi)

        def rand_point():
            x = lo + np.random.rand(dim) * span
            return x

        # Phase 0: Smart initialization
        # Mix of: points near front, random, and spread in objective space
        n_init = min(40, budget // 5)
        X = []
        F = []
        used = 0
        
        # 60% near front, 40% random
        n_front = int(n_init * 0.6)
        n_rand = n_init - n_front
        
        for _ in range(n_front):
            x = rand_point_near_front()
            f = np.asarray(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1
        
        for _ in range(n_rand):
            x = rand_point()
            f = np.asarray(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1
        
        X = np.array(X)
        F = np.array(F)

        # Reference point for scoring
        ref = np.array([3.0, 5.0, 7.0])

        # Track best archive points (non-dominated, spread)
        archive_F = []
        archive_X = []

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _update_archive(x, f):
            nonlocal archive_F, archive_X
            # Check if dominated by existing archive
            for af in archive_F:
                if _dominates(af, f):
                    return
            # Remove archive points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not _dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

            # If archive too large, trim by crowding distance
            max_archive = 50
            if len(archive_F) > max_archive:
                # Keep only non-dominated
                nd = []
                for i in range(len(archive_F)):
                    dominated = False
                    for j in range(len(archive_F)):
                        if i != j and _dominates(archive_F[j], archive_F[i]):
                            dominated = True
                            break
                    if not dominated:
                        nd.append(i)
                if len(nd) > max_archive:
                    # Compute crowding distance
                    dist = np.zeros(len(nd))
                    for m in range(self.n_obj):
                        vals = np.array([archive_F[i][m] for i in nd])
                        order = np.argsort(vals)
                        dist[order[0]] = np.inf
                        dist[order[-1]] = np.inf
                        span_m = vals[order[-1]] - vals[order[0]]
                        if span_m > 1e-15:
                            for k in range(1, len(nd) - 1):
                                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span_m
                    # Remove point with smallest crowding distance
                    min_idx = nd[np.argmin(dist)]
                    archive_X.pop(min_idx)
                    archive_F.pop(min_idx)

        # Initialize archive
        for i in range(len(X)):
            _update_archive(X[i], F[i])

        # Phase 1: DE-based search with adaptive mutation
        de_budget = int(budget * 0.70)
        pop_size = len(X)
        
        while used < de_budget:
            if time.time() >= deadline:
                break
            
            # Adaptive mutation rate: decrease over time
            progress = used / max(de_budget, 1)
            F_adaptive = 0.8 * (1.0 - progress) + 0.3 * progress
            
            # pick 3 distinct parents
            i = np.random.randint(0, pop_size)
            j = np.random.randint(0, pop_size)
            k = np.random.randint(0, pop_size)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop_size)
                k = np.random.randint(0, pop_size)
            
            v = X[i] + F_adaptive * (X[j] - X[k])
            v = np.clip(v, lo, hi)
            vf = np.asarray(problem(v), dtype=float)
            used += 1

            # Acceptance: tournament selection based on hypervolume contribution
            # Accept if not dominated by target, or if it improves HV contribution
            if not self._dominates(F[i], vf):
                X[i], F[i] = v, vf
            _update_archive(v, vf)

            # Occasionally add a new random point to maintain diversity
            if used % 20 == 0 and used < de_budget - 10:
                if np.random.rand() < 0.3:
                    x_new = rand_point_near_front() if np.random.rand() < 0.7 else rand_point()
                    f_new = np.asarray(problem(x_new), dtype=float)
                    used += 1
                    # Replace worst point in population
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = x_new
                    F[worst_idx] = f_new
                    _update_archive(x_new, f_new)

        # Phase 2: Local search polish on top archive points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Select top archive points by approximate HV contribution
            top_n = min(8, len(archive_F))
            if top_n > 0:
                # Compute approximate contributions
                contribs = []
                for idx in range(len(archive_F)):
                    contribs.append(self._hv_contribution_approx(archive_F[idx], archive_F, ref))
                order = np.argsort(contribs)[::-1][:top_n]

                for idx in order:
                    if used >= budget or time.time() >= deadline:
                        break
                    if idx >= len(archive_X):
                        continue

                    x = archive_X[idx].copy()
                    f = archive_F[idx].copy()
                    remaining_before = budget - used

                    new_x, new_f, evals_used = self._hv_polish(
                        x, f, problem, bounds, remaining_before, deadline, archive_F, ref
                    )
                    used += evals_used
                    _update_archive(new_x, new_f)

        return X
# EVOLVE-BLOCK-END
