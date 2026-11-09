import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming matching the evaluator's logic.
    """
    n = len(points)
    # Lexicographic sort: x first, then y
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        # Calculate angles
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute cross products for turns: cross(a, b, c) = (b-a) x (c-b)
        # We need:
        # 1. cross(p, q[i], q[j]) for the first turn
        # 2. cross(q[i], q[j], q[k]) for intermediate turns
        # 3. cross(q[i], q[j], p) for closing turn 1
        # 4. cross(q[j], p, q[i]) for closing turn 2 (note order)

        # Precompute cross(q[i], q[j], q[k]) for all i < j < k
        # This is a 3D tensor but m is small (<=32).
        # Let's store cross_q[i, j, k] = cross(q[i], q[j], q[k])
        # Actually, we only need it in the DP loop. Let's compute on the fly or precompute a 2D matrix for fixed middle?
        # No, the middle changes.
        # Let's precompute cross_q[i, j, k] as a 3D array (m, m, m). 32^3 = 32768 floats. Trivial memory.
        
        cross_q = np.zeros((m, m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                v1x = q[j, 0] - q[i, 0]
                v1y = q[j, 1] - q[i, 1]
                for k in range(m):
                    if j == k:
                        continue
                    v2x = q[k, 0] - q[j, 0]
                    v2y = q[k, 1] - q[j, 1]
                    cross_q[i, j, k] = v1x * v2y - v1y * v2x

        # cross_p[i, j] = cross(p, q[i], q[j])
        cross_p = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            v1x = q[i, 0] - p[0]
            v1y = q[i, 1] - p[1]
            for j in range(m):
                v2x = q[j, 0] - q[i, 0]
                v2y = q[j, 1] - q[i, 1]
                cross_p[i, j] = v1x * v2y - v1y * v2x

        # cross_close1[i, j] = cross(q[i], q[j], p)
        cross_c1 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            v1x = q[j, 0] - q[i, 0] # Wait, v1 is q[j]-q[i]? No.
            # cross(a, b, c) = (b-a) x (c-b)
            # a=q[i], b=q[j], c=p
            # v1 = q[j] - q[i]
            # v2 = p - q[j]
            for j in range(m):
                v1x = q[j, 0] - q[i, 0]
                v1y = q[j, 1] - q[i, 1]
                v2x = p[0] - q[j, 0]
                v2y = p[1] - q[j, 1]
                cross_c1[i, j] = v1x * v2y - v1y * v2x

        # cross_close2[i, j] = cross(q[j], p, q[i])
        # a=q[j], b=p, c=q[i]
        # v1 = p - q[j]
        # v2 = q[i] - p
        cross_c2 = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                v1x = p[0] - q[j, 0]
                v1y = p[1] - q[j, 1]
                v2x = q[i, 0] - p[0]
                v2y = q[i, 1] - p[1]
                cross_c2[i, j] = v1x * v2y - v1y * v2x

        # DP: dp[l, s, i, j] where chain is p, q[s], ..., q[i], q[j]
        # l is length of chain (number of q points).
        # We want l=6.
        # dp[2, s, s, j] = 1 if cross_p[s, j] > 1e-9 else 0
        # dp[l, s, i, j] = sum_{k < i} dp[l-1, s, k, i] * I[cross_q[k, i, j] > 1e-9]
        
        # Dimensions: l up to 6, s up to m, i up to m, j up to m.
        # m <= 32. 6 * 32 * 32 * 32 = 196,608.
        dp = np.zeros((7, m, m, m), dtype=np.int64)

        # Init l=2
        for s in range(m):
            for j in range(s + 1, m):
                if cross_p[s, j] > 1e-9:
                    dp[2, s, s, j] = 1

        # Fill l=3..6
        for l in range(3, 7):
            for s in range(m):
                for i in range(s + 1, m):
                    for j in range(i + 1, m):
                        # Sum over k < i
                        # dp[l-1, s, k, i]
                        # We need cross_q[k, i, j] > 1e-9
                        # Note: k must be < i.
                        # Also k >= s? Yes, because chain starts at s and indices are increasing.
                        val = 0
                        for k in range(s, i):
                            if dp[l-1, s, k, i] > 0:
                                if cross_q[k, i, j] > 1e-9:
                                    val += dp[l-1, s, k, i]
                        dp[l, s, i, j] = val

        # Sum up l=6
        for s in range(m):
            for i in range(s + 1, m):
                for j in range(i + 1, m):
                    count = dp[6, s, i, j]
                    if count == 0:
                        continue
                    # Check closing turns
                    if cross_c1[i, j] > 1e-9 and cross_c2[s, j] > 1e-9:
                        total += count

    return total


def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False
    
    # Check distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False
        
    # Check collinearity
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True


def erdos_szekeres_problem7() -> np.ndarray:
    start = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start)

    # Strategy: Start with points on a circle, then use local search.
    # The circle is a good starting point because it has a known structure.
    # However, random points can sometimes be better if we can optimize them.
    
    # Initial: Circle
    angles = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    radius = 0.45
    cx, cy = 0.5, 0.5
    best_pts = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    best_pts += rng.normal(0, 0.01, (33, 2))
    best_pts = np.clip(best_pts, 0.01, 0.99)
    
    if not _is_valid_fast(best_pts):
        best_pts = rng.random((33, 2)) * 0.85 + 0.075
        best_pts += rng.normal(0, 0.005, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)
    
    best_c = _count_convex7_fast(best_pts)
    
    # Local search
    no_improve_count = 0
    max_no_improve = 1000
    
    while time_left() > 30 and no_improve_count < max_no_improve:
        improved = False
        
        # Try moving 1-2 points
        for _ in range(100):
            if time_left() < 30:
                break
            n_move = rng.integers(1, 3)
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                new_pts[i] = rng.random(2) * 0.85 + 0.075
            
            if not _is_valid_fast(new_pts):
                continue
            
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve_count = 0
                break
        
        # Try small perturbations
        if not improved:
            for _ in range(100):
                if time_left() < 30:
                    break
                i = rng.integers(0, 33)
                new_pts = best_pts.copy()
                new_pts[i] += rng.normal(0, 0.02, 2)
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
                
                if not _is_valid_fast(new_pts):
                    continue
                
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            # Occasionally try a larger jump
            if no_improve_count % 50 == 0:
                n_move = rng.integers(5, 15)
                idx = rng.choice(33, size=n_move, replace=False)
                new_pts = best_pts.copy()
                for i in idx:
                    new_pts[i] = rng.random(2) * 0.85 + 0.075
                
                if _is_valid_fast(new_pts):
                    c = _count_convex7_fast(new_pts)
                    if c < best_c:
                        best_c = c
                        best_pts = new_pts
                        no_improve_count = 0

    return best_pts

# EVOLVE-BLOCK-END
