# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    This matches the evaluator's logic:
    1. Sort points lexicographically.
    2. For each point p (pivot), consider points q with index > i.
    3. Sort q by angle around p.
    4. Count convex chains of length 6 starting from p.
    A convex 7-gon corresponds to a convex chain of 6 points q0...q5 such that
    the turn at p (closing the loop) is also convex.
    """
    n = len(points)
    # Lexicographic sort: x first, then y
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    # k=7 means we need chains of length 6 (6 points after pivot)
    # dp[l][j] = number of convex chains of length l ending at point j (in the angle-sorted list of later points)
    # The chain starts from the pivot p.
    
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
        # Sort indices by angle
        idx = np.argsort(angles)
        q = later[idx]
        
        # Precompute cross products for turns
        # turn(a, b, c) = (b-a) x (c-b)
        # We need turns for:
        # 1. (p, q[a], q[b])
        # 2. (q[a], q[b], q[c])
        
        # dp[l][j] where j is index in q (0..m-1)
        # l is chain length (number of points in q included so far)
        # We want chains of length 6.
        # dp[1][j] = 1 for all j (chain of just q[j])
        # dp[l][j] = sum_{i < j} dp[l-1][i] * I[turn(q[i-1], q[i], q[j]) > 0] 
        # Wait, the turn condition depends on the previous point in the chain.
        # Let's define dp[l][j] as number of valid chains of length l ending at q[j].
        # For l=1: dp[1][j] = 1.
        # For l>1: dp[l][j] = sum_{i < j} dp[l-1][i] * I[turn(prev, q[i], q[j]) > 0]
        # This requires knowing 'prev'. So we need dp[l][i][j]? No, that's O(m^2) state.
        # Standard DP for convex polygons:
        # Let dp[l][j] be the number of convex chains of length l ending at point j (among the later points).
        # The chain is p, q[i0], q[i1], ..., q[i_{l-1}=j].
        # The condition for adding q[j] to a chain ending at q[i] (length l-1) is that the turn (q[i_{l-2}], q[i], q[j]) is left.
        # This implies we need to track the previous point.
        
        # Correct DP:
        # dp[l][i][j] is too big? m is up to 32. 32^2 * 6 is small.
        # But we can optimize.
        # Let's use the property that points are sorted by angle.
        # A chain p, q[a0], q[a1], ..., q[a_{l-1}] is convex if all turns are left.
        # Turn (p, q[a0], q[a1]) > 0
        # Turn (q[a0], q[a1], q[a2]) > 0
        # ...
        
        # dp[l][j] = number of convex chains of length l ending at q[j].
        # To compute dp[l][j], we sum over i < j such that the turn (q[i_prev], q[i], q[j]) is left.
        # This depends on i_prev. So we must store dp[l][i][j] or similar.
        # Actually, the standard recurrence is:
        # dp[1][j] = 1
        # dp[l][j] = sum_{i < j} dp[l-1][i] * I[turn(q[i], q[j], q[k])?] No.
        
        # Let's look at the geometry.
        # If points are sorted by angle around p, then for any i < j < k,
        # the turn (q[i], q[j], q[k]) is left if and only if q[j] is to the left of vector q[i]->q[k]?
        # No, simpler:
        # The turn (a, b, c) is left if cross(b-a, c-b) > 0.
        
        # Let's just use the O(m^3 * k) or O(m^2 * k) DP.
        # dp[l][j] is not sufficient because the validity of adding j to a chain ending at i depends on the point before i.
        # So we need dp[l][i][j] = number of chains of length l ending with ..., i, j.
        # dp[1][j] is not defined in this pair form.
        # Let's define dp[l][j] as number of chains of length l ending at j, BUT we need to know the previous point to check the turn.
        
        # Alternative:
        # dp[l][j] = number of convex chains of length l ending at q[j].
        # For l=1: dp[1][j] = 1.
        # For l=2: dp[2][j] = number of i < j such that turn(p, q[i], q[j]) > 0.
        # For l>2: dp[l][j] = sum_{i < j} (number of chains of length l-1 ending at i such that the turn (prev, q[i], q[j]) > 0).
        # This still requires knowing 'prev'.
        
        # Let's use the full state: dp[l][i][j] where i < j.
        # dp[2][i][j] = 1 if turn(p, q[i], q[j]) > 0 else 0.
        # dp[l][i][j] = sum_{k < i} dp[l-1][k][i] * I[turn(q[k], q[i], q[j]) > 0].
        
        # Finally, a convex 7-gon corresponds to a chain of length 6: p, q[a0], ..., q[a5].
        # The turns inside the chain are convex by definition of the DP.
        # We must also check the closing turns:
        # 1. Turn (q[a4], q[a5], p) > 0
        # 2. Turn (q[a5], p, q[a0]) > 0
        
        # So for each chain ending at (a4, a5) with length 6, we check the closing conditions.
        
        # Dimensions: m up to 32. l up to 6.
        # dp[l] is a (m, m) matrix.
        # dp[2][i][j] for i < j.
        
        dp = np.zeros((7, m, m), dtype=np.int64)
        
        # Initialize dp[2]
        # turn(p, q[i], q[j]) = cross(q[i]-p, q[j]-q[i])
        # = (q[i][0]-p[0])*(q[j][1]-q[i][1]) - (q[i][1]-p[1])*(q[j][0]-q[i][0])
        
        # Vectorize dp[2] initialization
        # We need cross(p, q[i], q[j]) for all i < j.
        # Let's compute a matrix M2 where M2[i][j] = 1 if turn(p, q[i], q[j]) > 1e-9 else 0.
        # q is (m, 2). p is (2,).
        # q_i = q[i], q_j = q[j].
        # v1 = q_i - p
        # v2 = q_j - q_i
        # cross = v1[0]*v2[1] - v1[1]*v2[0]
        
        # To avoid O(m^2) python loops, we can compute this with broadcasting if m is small, or just loop since m<=32.
        # 32*32 = 1024 operations. Trivial.
        
        for i in range(m):
            for j in range(i + 1, m):
                v1x = q[i, 0] - p[0]
                v1y = q[i, 1] - p[1]
                v2x = q[j, 0] - q[i, 0]
                v2y = q[j, 1] - q[i, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > 1e-9:
                    dp[2, i, j] = 1
        
        # Fill dp[l] for l=3..6
        for l in range(3, 7):
            # dp[l][i][j] = sum_{k < i} dp[l-1][k][i] * I[turn(q[k], q[i], q[j]) > 0]
            # We can precompute turn(q[k], q[i], q[j]) for all k<i<j?
            # Or just loop. m is small.
            for i in range(1, m):
                for j in range(i + 1, m):
                    # Check turn(q[i], q[j], ...) wait, the turn is (q[k], q[i], q[j]).
                    # The previous point in the chain ending at i is k.
                    # So we sum over k < i.
                    s = 0
                    for k in range(i):
                        if dp[l-1, k, i] == 0:
                            continue
                        # Check turn(q[k], q[i], q[j])
                        v1x = q[i, 0] - q[k, 0]
                        v1y = q[i, 1] - q[k, 1]
                        v2x = q[j, 0] - q[i, 0]
                        v2y = q[j, 1] - q[i, 1]
                        cross = v1x * v2y - v1y * v2x
                        if cross > 1e-9:
                            s += dp[l-1, k, i]
                    dp[l, i, j] = s
        
        # Now sum up valid 7-gons.
        # A 7-gon is defined by the chain p, q[a0], q[a1], q[a2], q[a3], q[a4], q[a5].
        # This corresponds to a state dp[6][a4][a5].
        # We need to check the closing turns:
        # 1. turn(q[a4], q[a5], p) > 0
        # 2. turn(q[a5], p, q[a0]) > 0
        # Note: a0 is NOT stored in dp[6][a4][a5]. We lost track of a0!
        
        # This is the problem with the standard "count chains" DP if we don't store the start.
        # However, for a convex polygon, the start point a0 is determined by the geometry?
        # No.
        
        # We need to modify the DP to track the start point a0.
        # dp[l][start][last] ?
        # dp[l][a0][aj] = number of chains starting at a0 and ending at aj.
        # dp[2][a0][aj] = 1 if turn(p, q[a0], q[aj]) > 0.
        # dp[l][a0][aj] = sum_{i < aj} dp[l-1][a0][qi] * I[turn(qi, q[aj-1]?, q[aj]) > 0].
        # Still need the previous point.
        
        # dp[l][a0][i][j] = number of chains starting at a0, ending at i, j.
        # Size: 6 * 32 * 32 * 32 = ~200k integers. Very fast.
        
        # Let's restart the DP with this state.
        dp_full = np.zeros((7, m, m, m), dtype=np.int64)
        
        # l=2: chain p, q[a0], q[a1]. Ends at (a0, a1).
        # dp_full[2, a0, a1, a1]? No, state is (start, prev, curr).
        # Let's define state as (start_idx, prev_idx, curr_idx).
        # Chain: p, q[start], ..., q[prev], q[curr].
        # Length l means l points from q are included.
        # For l=2: points are q[start], q[curr]. prev is start?
        # Let's say state (s, p_idx, c_idx) represents a chain of length l ending with q[p_idx], q[c_idx], starting with q[s].
        # For l=2: chain is p, q[s], q[c]. The turn is (p, q[s], q[c]).
        # dp[2, s, s, c] = 1 if turn(p, q[s], q[c]) > 0.
        
        for s in range(m):
            for c in range(s + 1, m):
                v1x = q[s, 0] - p[0]
                v1y = q[s, 1] - p[1]
                v2x = q[c, 0] - q[s, 0]
                v2y = q[c, 1] - q[s, 1]
                cross = v1x * v2y - v1y * v2x
                if cross > 1e-9:
                    dp_full[2, s, s, c] = 1
        
        for l in range(3, 7):
            # dp[l, s, p_idx, c_idx] = sum_{k < p_idx} dp[l-1, s, k, p_idx] * I[turn(q[k], q[p_idx], q[c_idx]) > 0]
            # Note: k must be < p_idx because angles are sorted and chain must be convex (monotonic angle).
            for s in range(m):
                # p_idx must be > s
                for p_idx in range(s + 1, m):
                    for c_idx in range(p_idx + 1, m):
                        s_val = 0
                        for k in range(s, p_idx):
                            val = dp_full[l-1, s, k, p_idx]
                            if val == 0:
                                continue
                            # Check turn(q[k], q[p_idx], q[c_idx])
                            v1x = q[p_idx, 0] - q[k, 0]
                            v1y = q[p_idx, 1] - q[k, 1]
                            v2x = q[c_idx, 0] - q[p_idx, 0]
                            v2y = q[c_idx, 1] - q[p_idx, 1]
                            cross = v1x * v2y - v1y * v2x
                            if cross > 1e-9:
                                s_val += val
                        dp_full[l, s, p_idx, c_idx] = s_val
        
        # Sum up results for l=6
        # Chain: p, q[s], ..., q[p_idx], q[c_idx].
        # We need to check closing turns:
        # 1. turn(q[p_idx], q[c_idx], p) > 0
        # 2. turn(q[c_idx], p, q[s]) > 0
        
        for s in range(m):
            for p_idx in range(s + 1, m):
                for c_idx in range(p_idx + 1, m):
                    count = dp_full[6, s, p_idx, c_idx]
                    if count == 0:
                        continue
                    
                    # Check closing 1: (q[p_idx], q[c_idx], p)
                    v1x = q[c_idx, 0] - q[p_idx, 0]
                    v1y = q[c_idx, 1] - q[p_idx, 1]
                    v2x = p[0] - q[c_idx, 0]
                    v2y = p[1] - q[c_idx, 1]
                    cross1 = v1x * v2y - v1y * v2x
                    if cross1 <= 1e-9:
                        continue
                    
                    # Check closing 2: (q[c_idx], p, q[s])
                    v1x = p[0] - q[c_idx, 0]
                    v1y = p[1] - q[c_idx, 1]
                    v2x = q[s, 0] - p[0]
                    v2y = q[s, 1] - p[1]
                    cross2 = v1x * v2y - v1y * v2x
                    if cross2 <= 1e-9:
                        continue
                    
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
    # d2[i,j] = (xi-xj)^2 + (yi-yj)^2
    # Use broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0) # ignore self
    if np.any(d2 < 1e-12):
        return False
        
    # Check collinearity
    # |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)| < 1e-9
    # This is 2 * area.
    # We can check all triples. 33C3 = 5456. Fast enough in numpy if vectorized?
    # Or just loop since it's small.
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

    # Start with points on a circle (near the boundary of the unit square)
    # This is known to have relatively few convex k-gons for small k
    angles = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    radius = 0.45
    cx, cy = 0.5, 0.5
    best_pts = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    
    # Add small random perturbation to break symmetries and ensure validity
    best_pts += rng.normal(0, 0.01, (33, 2))
    best_pts = np.clip(best_pts, 0.01, 0.99)
    
    if not _is_valid_fast(best_pts):
        # Fallback to random if circle fails
        best_pts = rng.random((33, 2)) * 0.85 + 0.075
        best_pts += rng.normal(0, 0.005, (33, 2))
        best_pts = np.clip(best_pts, 0.01, 0.99)
    
    best_c = _count_convex7_fast(best_pts)
    best_score = 1.0 / (1.0 + best_c)
    
    # Local search with multiple strategies
    no_improve_count = 0
    max_no_improve = 500
    
    while time_left() > 30 and no_improve_count < max_no_improve:
        improved = False
        
        # Strategy 1: Random restarts - move 1-3 points to random positions
        for _ in range(30):
            if time_left() < 30:
                break
            n_move = rng.integers(1, 4)  # Move 1 to 3 points
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
        
        # Strategy 2: Small perturbations
        if not improved:
            for _ in range(50):
                if time_left() < 30:
                    break
                i = rng.integers(0, 33)
                new_pts = best_pts.copy()
                # Small Gaussian perturbation
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
        
        # Strategy 3: Swap two points' positions
        if not improved:
            for _ in range(30):
                if time_left() < 30:
                    break
                i, j = rng.choice(33, size=2, replace=False)
                new_pts = best_pts.copy()
                new_pts[[i, j]] = new_pts[[j, i]]
                
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
            # Occasionally try a larger jump to escape local minima
            if no_improve_count % 100 == 0:
                n_move = rng.integers(5, 12)
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
