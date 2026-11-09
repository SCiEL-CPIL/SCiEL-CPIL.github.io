# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0


def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming over convex chains.
    Matches the evaluator's logic:
    1. Sort points lexicographically.
    2. For each pivot p (index i), consider points q with index > i.
    3. Sort q by atan2 angle around p.
    4. Count convex chains of length 6 (6 points after pivot) such that
       the closing turn back to p is also convex.
    A turn p->q->r is convex if cross(q-p, r-q) > 1e-9.
    """
    n = len(points)
    # Lexicographic sort: x first, then y
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    k = 7  # we need 7-gons, so chains of length 6 after pivot

    for i in range(n):
        # Points after i in lexicographic order
        if n - i - 1 < k - 1:
            break
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        # Compute angles from p to each later point
        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        # Sort by angle
        angle_order = np.argsort(angles)
        q_sorted = later[angle_order]
        m = len(q_sorted)
        if m < k - 1:
            continue

        # dp[l][j] = number of convex chains of length l ending at point j
        # l ranges from 1 to k-1 = 6
        # j ranges from 0 to m-1
        # A chain of length l ending at j means we have l points q_{j1}, ..., q_{jl} = q_j
        # such that each consecutive turn is convex, and the turn from q_{j_{l-1}} to q_j
        # is convex. But we also need the turn from p to q_{j1} to be convex? 
        # Actually, the chain starts at p, then q_{j1}, then q_{j2}, ..., then q_{jl}.
        # The turns are: p->q_{j1}->q_{j2}, q_{j1}->q_{j2}->q_{j3}, ..., q_{j_{l-2}}->q_{j_{l-1}}->q_{jl}
        # And the closing turn: q_{jl}->q_{j_{l-1}}->p? No, the closing is q_{j_{l-1}}->q_{jl}->p.
        # Wait, the polygon is p, q_{j1}, q_{j2}, ..., q_{jl}. The turns are:
        # p->q_{j1}->q_{j2}, q_{j1}->q_{j2}->q_{j3}, ..., q_{j_{l-2}}->q_{j_{l-1}}->q_{jl},
        # and the closing turn q_{jl}->q_{j_{l-1}}->p? No, the last turn is q_{j_{l-1}}->q_{jl}->p.
        # Actually, for a polygon with vertices v0, v1, ..., v_{l}, the turns are at v1, v2, ..., v_l, v0.
        # So for p, q1, q2, ..., q6, the turns are at q1, q2, q3, q4, q5, q6, p.
        # Turn at q_i: cross(q_i - q_{i-1}, q_{i+1} - q_i) > 1e-9
        # Turn at p: cross(p - q6, q1 - p) > 1e-9

        # Let's redefine: dp[l][j] = number of convex chains of length l (l points after p)
        # ending at point j, where the chain is q_{j1}, ..., q_{jl} = q_j, and all turns
        # from p->q_{j1}->q_{j2} up to q_{j_{l-2}}->q_{j_{l-1}}->q_{jl} are convex.
        # We don't check the closing turn yet.

        # Initialize: dp[1][j] = 1 for all j (chain of length 1, just q_j, no turns to check yet? 
        # Actually, the first turn is p->q_{j1}->q_{j2}, so we need at least 2 points to have a turn.
        # Let's say dp[1][j] = 1 for all j, representing a chain of 1 point.
        # dp[2][j] = sum over i < j of dp[1][i] * (turn p->q_i->q_j is convex)
        # dp[l][j] = sum over i < j of dp[l-1][i] * (turn q_i->q_j->q_{next}?) 
        # Wait, the turn at q_{j_{l-1}} is cross(q_{j_{l-1}} - q_{j_{l-2}}, q_{j_l} - q_{j_{l-1}}).
        # So dp[l][j] = sum over i < j of dp[l-1][i] * (cross(q_i - q_{prev}, q_j - q_i) > 1e-9)
        # But we don't know q_{prev} in dp[l-1][i]. 

        # Let me re-read the evaluator's logic. The task says:
        # "points are sorted lexicographically, for each pivot the later points are sorted by atan2 angle around it,
        # and a turn p -> q -> r counts as convex only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9,
        # including the closing turns back to the pivot."

        # So the DP is: for each pivot p, sort later points by angle. Then count convex chains.
        # A convex chain of length L (L points) starting from p is a sequence q1, q2, ..., qL such that:
        # - p, q1, q2, ..., qL are in convex position (all turns convex)
        # - The turns are: p->q1->q2, q1->q2->q3, ..., q_{L-2}->q_{L-1}->q_L, and q_L->q_{L-1}->p
        # Wait, the closing turn is q_L->q_{L-1}->p? That doesn't seem right. The polygon is p, q1, q2, ..., q_L.
        # The turn at q_L is q_{L-1}->q_L->p. So the closing turn is q_{L-1}->q_L->p.

        # Let me define: dp[l][j] = number of convex chains of length l (l points after p) ending at q_j,
        # such that the turns p->q_{j1}->q_{j2}, q_{j1}->q_{j2}->q_{j3}, ..., q_{j_{l-2}}->q_{j_{l-1}}->q_{jl}
        # are all convex. We do NOT check the closing turn yet.
        # Then the total number of convex 7-gons with pivot p is:
        # sum over j of dp[6][j] * (turn q_{j5}->q_j->p is convex)
        # where q_{j5} is the 5th point in the chain (the one before q_j).

        # But dp[l][j] doesn't store the previous point. So we need a different DP.

        # Let me use: dp[l][i][j] = number of convex chains of length l ending with ...->q_i->q_j,
        # where l >= 2, and all turns up to q_i->q_j are convex.
        # Then dp[2][i][j] = 1 if turn p->q_i->q_j is convex, else 0, for i < j.
        # dp[l][i][j] = sum over h < i of dp[l-1][h][i] * (turn q_h->q_i->q_j is convex)
        # Then the total for pivot p is:
        # sum over i < j of dp[6][i][j] * (turn q_i->q_j->p is convex)

        # This is O(m^3 * k) per pivot, which is too slow for m=32, k=6.
        # m^3 = 32768, times 6 = 196608, times 33 pivots = 6.5 million. That's fine.

        # Let me implement this.
        # dp[l][i][j] for l in 2..6, i in 0..m-2, j in i+1..m-1
        # We can use a 3D array or iterate.

        # Let's use a 2D array for each l, or just iterate.
        # For efficiency, let's use numpy.

        # Precompute cross products for turns.
        # cross(i, j, k) = cross(q_i, q_j, q_k) = (q_j - q_i) x (q_k - q_j)
        # = (q_j[0] - q_i[0]) * (q_k[1] - q_j[1]) - (q_j[1] - q_i[1]) * (q_k[0] - q_j[0])
        # We need this for:
        # - p->q_i->q_j: cross(p, q_i, q_j)
        # - q_h->q_i->q_j: cross(q_h, q_i, q_j)
        # - q_i->q_j->p: cross(q_i, q_j, p)

        # Let's precompute a 3D array cross[i][j][k] for all i, j, k in 0..m-1,
        # but that's m^3 = 32768, which is fine.
        # Actually, we only need cross for specific combinations. Let's compute on the fly.

        # For dp[2][i][j]: cross(p, q_i, q_j) > 1e-9
        # For dp[l][i][j]: cross(q_h, q_i, q_j) > 1e-9 for h < i
        # For closing: cross(q_i, q_j, p) > 1e-9

        # Let's compute cross(p, q_i, q_j) for all i < j.
        # cross(p, q_i, q_j) = (q_i - p) x (q_j - q_i)
        # = (q_i[0] - p[0]) * (q_j[1] - q_i[1]) - (q_i[1] - p[1]) * (q_j[0] - q_i[0])

        # Let me define a helper function for cross.
        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

        # This is too slow in Python. Let me vectorize.
        # For dp[2][i][j]: we need cross(p, q_i, q_j) for all i < j.
        # Let's compute a matrix C2[i][j] = cross(p, q_i, q_j) for i < j.
        # q_i and q_j are 2D points.
        # cross(p, q_i, q_j) = (q_i[0] - p[0]) * (q_j[1] - q_i[1]) - (q_i[1] - p[1]) * (q_j[0] - q_i[0])
        # Let's vectorize over j for each i.

        # Actually, let's just use numpy for the whole DP.
        # m is at most 32, so we can afford O(m^3 * k).

        # Let's use a 3D array dp of shape (k, m, m), but only use dp[l][i][j] for i < j.
        # k = 6, m = 32, so 6*32*32 = 6144, which is tiny.

        dp = np.zeros((k, m, m), dtype=np.int64)
        # dp[2][i][j] for i < j
        for i in range(m - 1):
            for j in range(i + 1, m):
                c = cross(p, q_sorted[i], q_sorted[j])
                if c > 1e-9:
                    dp[2][i][j] = 1

        for l in range(3, k + 1):
            for i in range(m - 1):
                for j in range(i + 1, m):
                    # dp[l][i][j] = sum over h < i of dp[l-1][h][i] * (cross(q_h, q_i, q_j) > 1e-9)
                    s = 0
                    for h in range(i):
                        if dp[l - 1][h][i] > 0:
                            c = cross(q_sorted[h], q_sorted[i], q_sorted[j])
                            if c > 1e-9:
                                s += dp[l - 1][h][i]
                    dp[l][i][j] = s

        # Now sum over i < j of dp[6][i][j] * (cross(q_i, q_j, p) > 1e-9)
        for i in range(m - 1):
            for j in range(i + 1, m):
                if dp[k][i][j] > 0:
                    c = cross(q_sorted[i], q_sorted[j], p)
                    if c > 1e-9:
                        total += dp[k][i][j]

    return total


def _is_valid_fast(points):
    """
    Check validity:
    - Shape (33, 2)
    - No NaN or inf
    - All coordinates in [-1e-6, 1 + 1e-6]
    - No two points with squared distance < 1e-12
    - No three points with |cross| < 1e-9
    """
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check pairwise distances
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            if dx * dx + dy * dy < 1e-12:
                return False
    # Check collinearity
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross_val = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if abs(cross_val) < 1e-9:
                    return False
    return True


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points in the unit square
    to minimize the number of convex 7-gons.
    """
    start_time = time.time()
    n = 33

    # Start with a random configuration
    np.random.seed(42)
    best_points = np.random.rand(n, 2)
    best_count = _count_convex7_fast(best_points)
    best_valid = _is_valid_fast(best_points)

    # Local search
    step_size = 0.05
    max_iter = 5000

    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - _SAFE_MARGIN:
            break

        # Pick a random point to perturb
        idx = np.random.randint(n)
        orig = best_points[idx].copy()
        # Perturb by a small random amount
        best_points[idx] += np.random.randn(2) * step_size
        # Clamp to [0, 1]
        best_points[idx] = np.clip(best_points[idx], 0, 1)

        # Check validity
        if not _is_valid_fast(best_points):
            best_points[idx] = orig
            continue

        # Count convex 7-gons
        new_count = _count_convex7_fast(best_points)

        if new_count < best_count:
            best_count = new_count
            best_valid = True
            # Reduce step size
            step_size *= 0.95
        else:
            best_points[idx] = orig
            # Occasionally increase step size if stuck
            if it % 100 == 0:
                step_size = min(0.1, step_size * 1.1)

    # Final validation
    if not _is_valid_fast(best_points):
        # Fallback to a valid configuration
        np.random.seed(0)
        best_points = np.random.rand(n, 2)
        # Ensure validity
        for _ in range(1000):
            if _is_valid_fast(best_points):
                break
            np.random.seed(np.random.randint(1000000))
            best_points = np.random.rand(n, 2)

    return best_points


# EVOLVE-BLOCK-END
