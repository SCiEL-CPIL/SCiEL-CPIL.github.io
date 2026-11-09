import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    # Distance check
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    # Collinearity check
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

def _count_convex7_fast(points):
    """
    Count convex 7-gons using the DP approach described in the task contract.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    eps = 1e-9

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute turn matrix: turn[a, b, c] = (q[b]-q[a]) x (q[c]-q[b])
        # for a < b < c in the angle-sorted order
        turn = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                for c in range(b + 1, m):
                    v2x = q[c, 0] - q[b, 0]
                    v2y = q[c, 1] - q[b, 1]
                    turn[a, b, c] = v1x * v2y - v1y * v2x

        # turn_p[s, e] = turn(p, q[s], q[e]) for s < e
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            for e in range(s + 1, m):
                v2x = q[e, 0] - q[s, 0]
                v2y = q[e, 1] - q[s, 1]
                turn_p[s, e] = v1x * v2y - v1y * v2x

        # DP: dp[l][s][e] = number of convex chains of l points from q (indices s..e)
        # such that p -> q[s] -> ... -> q[e] forms a convex chain
        # l = 2: p -> q[s] -> q[e], valid if turn_p[s, e] > eps
        # l > 2: dp[l][s][e] = sum_{m: s < m < e} dp[l-1][s][m] * I(turn[m-1, m, e] > eps)
        # Wait, the turn at q[m] is (q[m-1], q[m], q[e]) where m-1 is the point before m in the chain.
        # Actually, the state dp[l-1][s][m] represents chains ending at m. To extend to e,
        # we need the turn at m, which depends on the point before m in the chain.
        # So dp[l-1][s][m] is not sufficient; we need to know the predecessor.
        
        # Correct DP: dp[l][s][m][e] is too complex.
        # Instead, use: dp[l][s][e] where the chain is p, q[s], ..., q[e].
        # The turn at the last point q[e] when extending from q[m] depends on q[m-1], q[m], q[e].
        # So we need to track the previous point.
        
        # Let's use dp[l][s][m] = number of convex chains of length l (in q) starting at s, ending at m.
        # To extend to e > m: check turn(prev, m, e) where prev is the point before m.
        # But dp[l][s][m] aggregates over all possible prevs.
        
        # Alternative: dp[l][s][m] is the count, and we store it as a sum.
        # When extending to e, the turn at m is (m-1, m, e) in the chain.
        # But we don't know m-1 from dp[l][s][m].
        
        # So the state must be (s, m) for length l, and we need to know the predecessor.
        # This means we need dp[l][s][m] to be broken down by predecessor.
        # dp[l][s][m] = sum_{prev < m} count[l][s][prev, m]
        # where count[l][s][prev, m] = 1 if chain s, ..., prev, m is valid.
        
        # Let's use count[l][s][prev, m] for l >= 2.
        # count[2][s][s, m] = 1 if turn_p[s, m] > eps
        # count[l][s][prev, m] = count[l-1][s][prevprev, prev] * I(turn[prevprev, prev, m] > eps)
        # This is getting complicated.
        
        # For m <= 32, let's just do the O(m^4) enumeration for each pivot.
        # Total work: sum_{i} O(m_i^4) where m_i = n-1-i.
        # Sum_{m=6}^{32} m^4 ≈ 32^5/5 ≈ 3.3M. This is fine.
        
        # For each pivot p, enumerate all 6-tuples (a, b, c, d, e, f) with 0 <= a < b < c < d < e < f < m
        # Check if p, q[a], q[b], q[c], q[d], q[e], q[f] forms a convex 7-gon.
        # The turns are:
        # 1. turn_p[a, b] > eps
        # 2. turn[a, b, c] > eps
        # 3. turn[b, c, d] > eps
        # 4. turn[c, d, e] > eps
        # 5. turn[d, e, f] > eps
        # 6. turn(q[f], p, q[a]) > eps  (closing turn at f: (q[f]-q[e]) x (p-q[f]))
        # 7. turn(p, q[a], q[b]) is already #1, but the closing is at p: (p-q[f]) x (q[a]-p)
        # Wait, the 7 turns are at each vertex of the 7-gon.
        # The 7-gon is p, q[a], q[b], q[c], q[d], q[e], q[f].
        # Turns:
        # At p: (q[a]-p) x (q[f]-p) ... no, the turn at p is from q[f] to q[a].
        # The turn at vertex v_i is (v_{i+1} - v_i) x (v_{i+2} - v_{i+1})? No.
        # The turn at v_i is (v_i - v_{i-1}) x (v_{i+1} - v_i).
        
        # Let's define the 7 points in order: v0=p, v1=q[a], v2=q[b], v3=q[c], v4=q[d], v5=q[e], v6=q[f].
        # Turn at v0: (v1-v0) x (v6-v0)? No, the turn at v0 is (v0-v6) x (v1-v0).
        # Actually, for a convex polygon, all cross products (v_{i+1}-v_i) x (v_{i+2}-v_{i+1}) should have the same sign.
        # But the task says: "a turn p -> q -> r counts as convex only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9"
        # This is the cross product of (q-p) and (r-q), i.e., (q-p) x (r-q).
        # So the turn at q in the sequence p, q, r is (q-p) x (r-q).
        
        # For the 7-gon p, q[a], q[b], q[c], q[d], q[e], q[f]:
        # Turn at q[a]: (q[a]-p) x (q[b]-q[a]) = turn_p[a, b]
        # Turn at q[b]: (q[b]-q[a]) x (q[c]-q[b]) = turn[a, b, c]
        # Turn at q[c]: (q[c]-q[b]) x (q[d]-q[c]) = turn[b, c, d]
        # Turn at q[d]: (q[d]-q[c]) x (q[e]-q[d]) = turn[c, d, e]
        # Turn at q[e]: (q[e]-q[d]) x (q[f]-q[e]) = turn[d, e, f]
        # Turn at q[f]: (q[f]-q[e]) x (p-q[f])
        # Turn at p: (p-q[f]) x (q[a]-p)
        
        # All 7 must be > eps.
        
        for a in range(m - 5):
            if turn_p[a, a+1] <= eps:
                continue
            for b in range(a + 1, m - 4):
                if turn[a, b, b+1] <= eps:
                    continue
                for c in range(b + 1, m - 3):
                    if turn[b, c, c+1] <= eps:
                        continue
                    for d in range(c + 1, m - 2):
                        if turn[c, d, d+1] <= eps:
                            continue
                        for e in range(d + 1, m - 1):
                            if turn[d, e, e+1] <= eps:
                                continue
                            for f in range(e + 1, m):
                                if turn[e, f, f] <= eps:  # placeholder, need turn at f
                                    continue
                                # Turn at q[f]: (q[f]-q[e]) x (p-q[f])
                                v1x = q[f, 0] - q[e, 0]
                                v1y = q[f, 1] - q[e, 1]
                                v2x = p[0] - q[f, 0]
                                v2y = p[1] - q[f, 1]
                                cross_f = v1x * v2y - v1y * v2x
                                if cross_f <= eps:
                                    continue
                                # Turn at p: (p-q[f]) x (q[a]-p)
                                v1x = p[0] - q[f, 0]
                                v1y = p[1] - q[f, 1]
                                v2x = q[a, 0] - p[0]
                                v2y = q[a, 1] - p[1]
                                cross_p = v1x * v2y - v1y * v2x
                                if cross_p <= eps:
                                    continue
                                total += 1

    return total

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Initial points: 4 clusters at corners
    pts = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        for _ in range(8):
            x = cx + rng.normal(0, 0.02)
            y = cy + rng.normal(0, 0.02)
            pts.append((x, y))
    # 1 remaining point
    x = rng.uniform(0.45, 0.55)
    y = rng.uniform(0.45, 0.55)
    pts.append((x, y))
    
    best_pts = np.array(pts)
    best_pts = np.clip(best_pts, 0.01, 0.99)
    
    # Ensure validity
    if not _is_valid_fast(best_pts):
        for _ in range(100):
            best_pts += rng.normal(0, 0.005, (33, 2))
            best_pts = np.clip(best_pts, 0.01, 0.99)
            if _is_valid_fast(best_pts):
                break
                
    best_c = _count_convex7_fast(best_pts)
    
    step_size = 0.05
    no_improve = 0
    
    while time_left() > 5:
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        new_pts[i] = best_pts[i] + rng.normal(0, step_size, 2)
        new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                no_improve = 0
                step_size = min(0.05, step_size * 1.1)
            else:
                no_improve += 1
                if no_improve > 50:
                    step_size = max(0.005, step_size * 0.9)
                    no_improve = 0
    
    return best_pts
