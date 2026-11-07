"""Multi-objective optimizer on WFG9: NSGA-II + Hypervolume replacement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _dominance_matrix(F):
    """Return bool matrix D where D[i,j] is True if i dominates j."""
    n = F.shape[0]
    # i dominates j if i <= j in all objectives and i < j in at least one
    le = (F[:, None, :] <= F[None, :, :])  # (n, n, m)
    lt = (F[:, None, :] < F[None, :, :])
    D = le.all(axis=2) & lt.any(axis=2)
    return D


def _fast_non_dominated_sort(F):
    """Return list of fronts (each a list of indices)."""
    n = F.shape[0]
    D = _dominance_matrix(F)
    S = [set() for _ in range(n)]   # S[i] = set of indices dominated by i
    nd = np.zeros(n, dtype=int)     # nd[i] = count of indices dominating i
    for i in range(n):
        for j in range(n):
            if D[i, j]:
                S[i].add(j)
            if D[j, i]:
                nd[i] += 1
    fronts = []
    front = [i for i in range(n) if nd[i] == 0]
    while front:
        fronts.append(front)
        nxt = []
        for i in front:
            for j in S[i]:
                nd[j] -= 1
                if nd[j] == 0:
                    nxt.append(j)
        front = nxt
    return fronts


def _crowding_distance(F, idx):
    """Crowding distance for indices idx."""
    m = F.shape[1]
    dist = np.zeros(len(idx))
    if len(idx) <= 2:
        dist[:] = np.inf
        return dist
    for k in range(m):
        vals = F[idx, k]
        mn, mx = vals.min(), vals.max()
        if mx - mn < 1e-15:
            continue
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        for t in range(1, len(order) - 1):
            dist[order[t]] += (vals[order[t + 1]] - vals[order[t - 1]]) / (mx - mn)
    return dist


def _hv_contribution_3d(points, ref):
    """Compute hypervolume contribution for 3D points vs reference (all points better than ref).
    Returns total hypervolume. Simple O(n^3) for small n, or sort-based for larger."""
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float)
    # filter: strictly better in all objectives
    mask = np.all(pts < ref, axis=1)
    pts = pts[mask]
    n = len(pts)
    if n == 0:
        return 0.0
    # For 3D, use the standard algorithm: sort by f1, then for each point compute
    # the 2D hypervolume of its "shadow"
    # Simpler: use inclusion-exclusion via sorting
    # We'll use the approach: sort by obj0, maintain the 2D HV of the remaining region
    # Actually for small n (<= 200), a simple approach:
    # HV = sum over non-dominated points of their contribution
    # For 3D, contribution of point p (after sorting by f0) is:
    #   (ref0 - p0) * HV_2D of the 2D points (p1, p2) restricted to (p1, p2) < (ref1, ref2)
    #   minus what's already covered
    # Let's use a cleaner O(n^2 log n) for 3D
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    # For each point in sorted order, its contribution in 3D:
    # width in dim0 = ref[0] - pts[i,0]
    # times the 2D hypervolume of the "envelope" of points j with pts[j,1] > pts[i,1] and pts[j,2] > pts[i,2]
    # This is getting complex. Let me use a simpler but correct O(n^3) for n<=200
    # Actually n can be up to 200, so O(n^3) = 8e6 which is fine
    total = 0.0
    # For each point, compute the box it uniquely covers
    # A point p covers the box [p, ref] minus what's covered by other points
    # In 3D, we can compute: for point i, the region where i is the "closest" in all dims
    # Simpler correct approach: inclusion-exclusion is hard. Use the standard 3D HV:
    # Sort by f0. For point i, its contribution is (ref0-fi0) * HV2D({(fj1,fj2) for j where fj1>fi1 and fj2>fi2} U {(ref1,ref2)})
    # where HV2D is the 2D hypervolume of the set of points vs reference (ref1,ref2)
    for i in range(n):
        # 2D HV of points that are "beyond" point i in dims 1,2
        mask2 = (pts[:, 1] > pts[i, 1]) & (pts[:, 2] > pts[i, 2])
        pts2 = pts[mask2, 1:3]
        pts2 = np.vstack([pts2, [[ref[1], ref[2]]]])
        # 2D HV: sort by dim0, sum (ref1-x0)*(ref2-y)
        if len(pts2) == 1:
            hv2 = (ref[1] - pts2[0, 0]) * (ref[2] - pts2[0, 1])
        else:
            pts2 = pts2[np.argsort(pts2[:, 0])]
            # Keep only Pareto-optimal in 2D (minimize both)
            keep = [0]
            for j in range(1, len(pts2)):
                if pts2[j, 1] < pts2[keep[-1], 1]:
                    keep.append(j)
            pts2k = pts2[keep]
            hv2 = 0.0
            for j in range(len(pts2k)):
                x_next = pts2k[j + 1, 0] if j + 1 < len(pts2k) else ref[1]
                hv2 += (x_next - pts2k[j, 0]) * (ref[2] - pts2k[j, 1])
        total += (ref[0] - pts[i, 0]) * hv2
    return total


class Optimizer:
    """NSGA-II with hypervolume-aware replacement for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s/330s limit
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size: use a good fraction of budget
        N = min(40, max(10, self.budget // 5))
        # Ensure we can do at least a few generations
        N = min(N, self.budget // 2)

        # Initial population: random
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # All evaluated points (for final HV)
        all_X = [X.copy()]
        all_F = [F.copy()]

        # NSGA-II loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_off = min(N, self.budget - used)

            for _ in range(n_off):
                if time.time() > deadline:
                    break

                # Selection: binary tournament on combined ranking
                # First, get fronts and crowding for current population
                fronts = _fast_non_dominated_sort(F)
                # Build rank array
                rank = np.zeros(N, dtype=int)
                for fi, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = fi
                # Crowding for each front
                cd = np.zeros(N)
                for front in fronts:
                    if len(front) > 2:
                        cd[front] = _crowding_distance(F, front)

                # Binary tournament
                i1 = rng.integers(N)
                i2 = rng.integers(N)
                if rank[i1] < rank[i2]:
                    p1 = i1
                elif rank[i2] < rank[i1]:
                    p1 = i2
                else:
                    p1 = i1 if cd[i1] > cd[i2] else i2

                i3 = rng.integers(N)
                i4 = rng.integers(N)
                if rank[i3] < rank[i4]:
                    p2 = i3
                elif rank[i4] < rank[i3]:
                    p2 = i4
                else:
                    p2 = i3 if cd[i3] > cd[i4] else i4

                # Crossover (SBX-like: blend)
                alpha = rng.random()
                child = alpha * X[p1] + (1 - alpha) * X[p2]

                # Adaptive mutation: larger early, smaller late
                progress = used / self.budget
                mut_scale = 0.3 * (1 - progress) + 0.02
                child = child + mut_scale * span * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent + offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            n_comb = len(X_comb)

            # Fast non-dominated sort on combined
            fronts = _fast_non_dominated_sort(F_comb)

            # Select next population of size N
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Need to select from this front using hypervolume contribution
                    remaining = N - len(new_X)
                    if remaining <= 0:
                        break
                    # Compute crowding distance for selection
                    front_F = F_comb[front]
                    cd = _crowding_distance(front_F, list(range(len(front))))
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for j in order[:remaining]:
                        idx = front[j]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)
            all_X.append(X.copy())
            all_F.append(F.copy())

        # Return (ignored by evaluator, but keep interface)
        return X
# EVOLVE-BLOCK-END
