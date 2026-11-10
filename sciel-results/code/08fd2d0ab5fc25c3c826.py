import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

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

def _count_convex7(points):
    """
    Count convex 7-gons using the DP described in the task contract.
    Points are sorted lexicographically. For each pivot i, later points are
    sorted by atan2 angle around it. A turn p->q->r is convex if
    (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
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

        # Precompute turn[a, b, c] = (q[b]-q[a]) x (q[c]-q[b]) for a < b < c
        turn = np.zeros((m, m, m), dtype=np.float64)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[b+1:, 0] - q[b, 0]
                v2y = q[b+1:, 1] - q[b, 1]
                turn[a, b, b+1:] = v1x * v2y - v1y * v2x

        # turn_p[s, e] = turn(p, q[s], q[e]) for s < e
        turn_p = np.zeros((m, m), dtype=np.float64)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[s+1:, 0] - q[s, 0]
            v2y = q[s+1:, 1] - q[s, 1]
            turn_p[s, s+1:] = v1x * v2y - v1y * v2x

        # dp[l][s][e] = number of convex chains of l points from q, starting at s, ending at e
        # such that p -> q[s] -> ... -> q[e] is a valid convex chain.
        # l ranges from 2 to 6 (total 7 points including p).
        
        # dp[2][s][e] = 1 if turn_p[s,e] > eps
        dp = np.zeros((7, m, m), dtype=np.int64)
        for s in range(m):
            valid_e = np.where(turn_p[s, s+1:] > eps)[0]
            if len(valid_e) > 0:
                dp[2, s, s+1:][valid_e] = 1

        # dp[l][s][e] = sum over mid in (s, e) where turn[mid_prev, mid, e] > eps
        # But we need the predecessor. So we use:
        # dp[l][s][e] = sum_{mid: s < mid < e} dp[l-1][s][mid] * I(turn at mid is convex w.r.t. e)
        # The turn at mid for chain p->...->mid->e is (q[mid-1_in_chain], q[mid], q[e])
        # This depends on the specific predecessor, so dp[l-1][s][mid] is insufficient.
        
        # Correct approach: dp[l][s][e] where the chain is p, q[s], ..., q[e]
        # The last turn is at the second-to-last point.
        # So we need to track the second-to-last point.
        
        # Let's use: dp[l][s][p_idx] = number of chains of length l ending at p_idx
        # This is still not enough. We need the predecessor.
        
        # Alternative: enumerate all 6-tuples from q (sorted by angle) and check.
        # C(32,6) ~ 906k per pivot, times 27 pivots ~ 24M. Too slow in Python.
        
        # Let's use the 4D DP: dp[l][s][p_idx][e] is too big.
        
        # Actually, for the closing turn check, we need:
        # turn(q[e], p, q[s]) > eps  (closing turn at p)
        # and turn(p, q[s], q[next]) > eps (first turn after p)
        
        # Let me just do the direct check for 6-tuples from q, but vectorize.
        
        # For each pivot i, we have m points in q.
        # We need to count 6-subsets of q such that:
        # 1. When sorted by angle around p, the turns are all convex
        # 2. The closing turn q[last] -> p -> q[first] is convex
        
        # This is complex. Let's use a simpler but correct approach:
        # For each 7-subset of the 33 points, check if it's convex.
        # C(33,7) = 4,292,145. For each, sort by angle around centroid and check 7 turns.
        # This is ~30M operations. In Python, this might take ~30-60s. Too slow.
        
        # Let's go back to the DP with proper state.
        # State: dp[l][s][e] = count of convex chains p -> q[s] -> ... -> q[e] of length l+1
        # (l points from q). The turn at each interior point is convex.
        # To extend from e to f (f > e in angle order), we need turn(q[e_prev], q[e], q[f]) > eps.
        # But dp[l][s][e] doesn't tell us e_prev.
        
        # So we need: dp[l][s][e] to be a dict or we need to include e_prev in the state.
        # dp[l][s][p_idx][e] where p_idx is the point before e. Size: 6*32*32*32 = 196k. OK.
        
        dp4 = np.zeros((7, m, m, m), dtype=np.int64)
        
        # Base: l=1 (just p -> q[s]). No turn to check.
        # l=2: p -> q[s] -> q[e]. Check turn_p[s,e] > eps.
        for s in range(m):
            for e in range(s+1, m):
                if turn_p[s, e] > eps:
                    dp4[2, s, s, e] = 1
        
        # l=3 to 6:
        # dp4[l][s][p_idx][e] = sum over mid in (s, p_idx) of dp4[l-1][s][mid][p_idx] * I(turn(q[mid], q[p_idx], q[e]) > eps)
        # Wait, the turn at p_idx when extending to e is (q[mid], q[p_idx], q[e]) where mid is the point before p_idx.
        
        for l in range(3, 7):
            for s in range(m):
                for p_idx in range(s+1, m):
                    for e in range(p_idx+1, m):
                        # Sum over all mid in (s, p_idx) such that dp4[l-1][s][mid][p_idx] > 0
                        # and turn(q[mid], q[p_idx], q[e]) > eps
                        mid_range = range(s, p_idx)
                        count = 0
                        for mid in mid_range:
                            c = dp4[l-1, s, mid, p_idx]
                            if c == 0:
                                continue
                            if turn[mid, p_idx, e] > eps:
                                count += c
                        dp4[l, s, p_idx, e] = count
        
        # Now for each valid chain of length 6 (6 points from q, total 7 with p),
        # check the closing turn: q[e] -> p -> q[s]
        for s in range(m):
            for p_idx in range(s+1, m):
                for e in range(p_idx+1, m):
                    c = dp4[6, s, p_idx, e]
                    if c == 0:
                        continue
                    # Closing turn: q[e] -> p -> q[s]
                    v1x = p[0] - q[e, 0]
                    v1y = p[1] - q[e, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross = v1x * v2y - v1y * v2x
                    if cross > eps:
                        total += c

    return total

def _make_cup_cap(n_upper, n_lower, rng):
    """
    Create a cup-cap configuration: n_upper points on an upper convex chain,
    n_lower points on a lower convex chain.
    """
    pts = []
    # Upper chain: from left to right, convex downward
    for i in range(n_upper):
        t = i / max(n_upper - 1, 1)
        x = 0.05 + 0.9 * t
        y = 0.95 - 0.3 * np.sin(np.pi * t)
        x += rng.normal(0, 0.01)
        y += rng.normal(0, 0.01)
        pts.append([x, y])
    
    # Lower chain: from right to left, convex upward
    for i in range(n_lower):
        t = i / max(n_lower - 1, 1)
        x = 0.95 - 0.9 * t
        y = 0.05 + 0.3 * np.sin(np.pi * t)
        x += rng.normal(0, 0.01)
        y += rng.normal(0, 0.01)
        pts.append([x, y])
    
    return np.array(pts)

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    best_pts = None
    best_c = float('inf')
    
    # Try several initial configurations
    configs = [
        (17, 16),
        (16, 17),
        (15, 18),
        (18, 15),
        (14, 19),
        (19, 14),
    ]
    
    for nu, nl in configs:
        pts = _make_cup_cap(nu, nl, rng)
        pts = np.clip(pts, 0.01, 0.99)
        
        if not _is_valid_fast(pts):
            for _ in range(50):
                pts += rng.normal(0, 0.003, (33, 2))
                pts = np.clip(pts, 0.01, 0.99)
                if _is_valid_fast(pts):
                    break
            if not _is_valid_fast(pts):
                continue
        
        c = _count_convex7(pts)
        if c < best_c:
            best_c = c
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random points
        best_pts = rng.uniform(0.05, 0.95, (33, 2))
        best_c = _count_convex7(best_pts)
    
    # Local search with simulated annealing
    current_pts = best_pts.copy()
    current_c = best_c
    T = 0.1
    T_min = 0.001
    
    while time_left() > 30:
        # Pick 1-3 points to move
        n_move = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
        indices = rng.choice(33, size=n_move, replace=False)
        
        new_pts = current_pts.copy()
        for idx in indices:
            new_pts[idx] += rng.normal(0, 0.03, 2)
        new_pts = np.clip(new_pts, 0.01, 0.99)
        
        if not _is_valid_fast(new_pts):
            continue
        
        new_c = _count_convex7(new_pts)
        delta_c = new_c - current_c
        
        if delta_c < 0 or rng.random() < np.exp(-delta_c / T):
            current_pts = new_pts
            current_c = new_c
            if new_c < best_c:
                best_c = new_c
                best_pts = new_pts.copy()
        
        T = max(T * 0.995, T_min)
        
        # Periodic restarts
        if time_left() > 60 and rng.random() < 0.01:
            # Restart from best with perturbation
            current_pts = best_pts + rng.normal(0, 0.05, (33, 2))
            current_pts = np.clip(current_pts, 0.01, 0.99)
            if _is_valid_fast(current_pts):
                current_c = _count_convex7(current_pts)
            T = 0.1
    
    return best_pts
