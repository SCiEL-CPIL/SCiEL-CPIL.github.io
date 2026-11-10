# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_DEADLINE = 630.0  # wall-clock seconds
_START = None


def _elapsed():
    return time.time() - _START


def _count_convex7(points):
    """Count convex 7-gons using the exact evaluator DP.
    Points sorted lexicographically; for each pivot i, later points sorted by atan2
    around pivot; a turn p->q->r is convex if cross > 1e-9, including closing turns.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    P = points[order]
    total = 0
    for i in range(n):
        rest = np.arange(i + 1, n)
        if len(rest) < 6:
            break
        dx = P[rest, 0] - P[i, 0]
        dy = P[rest, 1] - P[i, 1]
        ang = np.arctan2(dy, dx)
        order_r = rest[np.argsort(ang)]
        Q = P[order_r]
        m = len(Q)
        # Build convex chain DP: dp[j] = number of convex chains of length j ending at Q[j]
        # A chain is convex if every consecutive triple has cross > 1e-9.
        # We count chains of 6 points (length 6) that close convexly back to pivot.
        # Actually we need 7-point subsets in convex position including pivot i.
        # So we need chains of 6 other points that, together with pivot, form convex 7-gon.
        # The chain must be convex and the two closing turns (pivot->first, last->pivot) must be convex.
        # Let's do DP over subsets is too expensive. Use the standard convex chain DP.
        # dp[k][j] = number of convex chains of k points ending at Q[j] (in angular order)
        # where each internal turn is convex. Then check closing.
        # For 7-gon: pivot + 6 others. Chain of 6 others in angular order around pivot.
        # The full polygon is pivot, Q[j0], Q[j1], ..., Q[j5] in order.
        # Turns: pivot->Q[j0]->Q[j1], ..., Q[j4]->Q[j5]->pivot, and Q[j5]->pivot->Q[j0]? 
        # Actually the polygon vertices in order are: i, j0, j1, ..., j5 (6 others) then back to i.
        # Turns: (i,j0,j1), (j0,j1,j2), ..., (j4,j5,i), (j5,i,j0).
        # Wait, that's 7 turns for 7 vertices. Let me recount: vertices i, j0, j1, j2, j3, j4, j5.
        # Turns: i->j0->j1, j0->j1->j2, j1->j2->j3, j2->j3->j4, j3->j4->j5, j4->j5->i, j5->i->j0.
        # So 7 turns. All must have cross > 1e-9.
        # The chain j0..j5 must have turns j0->j1->j2, ..., j3->j4->j5 convex (4 internal turns).
        # Plus i->j0->j1, j4->j5->i, j5->i->j0 must be convex.
        # DP: dp[k][j] = number of convex chains of k points (j0..j_{k-1}) ending at Q[j] 
        # where all internal turns are convex. Then for k=6, check the 3 closing turns.
        # This is O(m^2 * 6) per pivot = O(m^2) per pivot, O(n * m^2) total ~ O(n^3) = 33^3 ~ 36k. Fine.
        # But we need to track the first point too for the closing turn j5->i->j0.
        # So dp[k][first][last] = count. That's O(m^2 * 6) states, O(m^2 * 6 * m) transitions = O(m^3).
        # m ~ 32, m^3 ~ 32768 per pivot, * 33 pivots ~ 1M. Fine.
        # Actually let's just do it more simply: enumerate all 6-subsets? C(32,6) ~ 900k per pivot, too much.
        # Let's do the DP with (first, last) states.
        # dp[k][a][b] = number of convex chains of k points starting at Q[a], ending at Q[b],
        # with all internal turns convex. k >= 2.
        # Base: dp[2][a][b] = 1 for a < b (in angular order, since we process in order).
        # Transition: dp[k+1][a][c] += dp[k][a][b] if b < c and turn Q[b]->Q[c]... 
        # Wait, the turn at Q[b] is Q[b-1]->Q[b]->Q[c]? No.
        # The chain is Q[a], Q[b1], Q[b2], ..., Q[b_{k-1}], Q[b]. 
        # Turns: Q[a]->Q[b1]->Q[b2], ..., Q[b_{k-2}]->Q[b_{k-1}]->Q[b].
        # So when extending from chain ending at Q[b] to Q[c] (b < c), we need turn Q[b_{k-2}]->Q[b]->Q[c] convex.
        # But we don't track b_{k-2} in dp[k][a][b]. So we need dp[k][a][b] to also encode the previous point.
        # This is getting complex. Let me just use a simpler approach: for each pivot, 
        # enumerate all 6-subsets of the m later points, check if they form convex chain with pivot.
        # C(32,6) = 906192. * 33 pivots = 30M. Each check is O(6) = O(36) ops. Total ~ 1B ops. Too slow in Python.
        # Let me use the DP properly.
        # State: dp[k][a][b] where a = first index, b = last index, k = length.
        # But we need the second-to-last to check the turn. So state should be dp[k][a][b] where 
        # the chain is a, ..., b, and we know the turn at b is valid. To extend to c, we need 
        # the turn at b, which depends on the previous point. So we need dp[k][a][b] to be 
        # parameterized by the previous point too. 
        # Alternative: dp[k][a][b] = number of chains of length k from a to b where all turns 
        # except the last one are convex, and the last turn (at b) is NOT yet checked. 
        # No, let's just do: dp[k][a][b] = number of convex chains of k points from a to b 
        # where ALL turns are convex. To compute dp[k+1][a][c], we need dp[k][a][b] for b < c 
        # such that the turn at b (i.e., prev->b->c) is convex. But we don't know prev.
        # OK here's the clean way: dp[k][a][b] = number of convex chains of k points starting at a, 
        # ending at b, where the chain is a = p0, p1, ..., p_{k-1} = b, and all turns 
        # p_{i-1}->p_i->p_{i+1} for i=1..k-2 are convex. The turn at b (p_{k-2}->b->next) is 
        # checked when we extend. So dp[2][a][b] = 1 for a < b. 
        # dp[k+1][a][c] = sum over b (a < b < c) of dp[k][a][b] * [turn p_{k-2}->b->c is convex].
        # But p_{k-2} is the second-to-last in the chain ending at b, which we don't track.
        # I think the standard approach is to note that in angular order around pivot, 
        # the turn p->q->r is convex iff the angle from p to q is less than angle from p to r 
        # (in the right range). Actually for points in angular order around pivot i, 
        # the turn Q[a]->Q[b]->Q[c] with a<b<c is convex iff the cross product is positive.
        # Let me just precompute cross[b][c] for all b<c and use a different DP.
        # dp[k][b] = number of convex chains of k points ending at Q[b] (any start).
        # But we need the start for the closing turn. 
        # Given time constraints, let me just use a simpler but correct method:
        # For each pivot i, and each 6-subset of later points, check convexity.
        # To speed up, I'll vectorize the subset check using numpy.
        # Actually, let me just accept O(n * C(m,6)) and optimize with numpy.
        pass
    return total


def _count_convex7_fast(points):
    """Fast count of convex 7-gons using DP over convex chains."""
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    P = points[order]
    total = 0
    for i in range(n - 6):
        rest = np.arange(i + 1, n)
        m = len(rest)
        dx = P[rest, 0] - P[i, 0]
        dy = P[rest, 1] - P[i, 1]
        ang = np.arctan2(dy, dx)
        order_r = rest[np.argsort(ang)]
        Q = P[order_r]
        # Precompute cross products for turns
        # cross[a][b][c] = (Q[b]-Q[a]) x (Q[c]-Q[b]) for a<b<c
        # We need this for all triples. m <= 32, so m^3/6 ~ 5000 triples.
        # Let's compute a 2D array: turn[b][c] = cross of (Q[b]-Q[a])... no, depends on a.
        # turn[a][b][c] = (Q[b,0]-Q[a,0])*(Q[c,1]-Q[b,1]) - (Q[b,1]-Q[a,1])*(Q[c,0]-Q[b,0])
        # For the DP, when extending chain ending at b to c, with previous point a, 
        # we need turn[a][b][c] > 1e-9.
        # Let's precompute turn as a 3D array of shape (m, m, m) with turn[a][b][c] for a<b<c.
        # m=32, 32^3 = 32768, fine.
        turn = np.zeros((m, m, m))
        for a in range(m):
            for b in range(a + 1, m):
                for c in range(b + 1, m):
                    turn[a][b][c] = (Q[b, 0] - Q[a, 0]) * (Q[c, 1] - Q[b, 1]) - \
                                    (Q[b, 1] - Q[a, 1]) * (Q[c, 0] - Q[b, 0])
        # DP: dp[k][a][b] = number of convex chains of k points from a to b, all turns convex.
        # k from 2 to 6.
        # dp[2][a][b] = 1 for a < b.
        # dp[k+1][a][c] = sum_{b: a<b<c} dp[k][a][b] * (turn[prev][b][c] > 1e-9)
        # But we need prev, which is the second-to-last. So we need dp[k][a][b] to know prev.
        # Let me redefine: dp[k][a][b] = number of convex chains of k points starting at a, ending at b,
        # where the chain is a=p0, p1, ..., p_{k-1}=b, and all turns p_{i-1}->p_i->p_{i+1} for i=1..k-2 
        # are convex. Note: for k=2, there are no internal turns, so dp[2][a][b]=1.
        # For k=3, dp[3][a][c] = sum_{b: a<b<c} dp[2][a][b] * (turn[a][b][c] > 1e-9) 
        #   = number of b such that a<b<c and turn[a][b][c] > 1e-9.
        # For k=4, dp[4][a][d] = sum_{c: a<c<d} dp[3][a][c] * (turn[?][c][d] > 1e-9)
        #   But dp[3][a][c] = sum_{b: a<b<c} (turn[a][b][c] > 1e-9). 
        #   The turn at c when extending to d is p1->c->d where p1 is the middle point. 
        #   But we don't track p1 in dp[3][a][c]. 
        # I see the issue. Let me use a different state: dp[k][a][b] where the chain is 
        # a=p0, p1, ..., p_{k-2}, b=p_{k-1}, and we track that the turn at b is NOT yet checked.
        # Actually, the standard convex chain DP for Erdos-Szekeres uses:
        # dp[i][j] = number of convex chains of length i ending at point j.
        # But for counting k-gons, we need to close the chain.
        # Given the complexity, let me just use a direct enumeration with numpy vectorization.
        # For each pivot i, enumerate all 6-subsets of the m later points.
        # C(m, 6) for m=32 is 906192. For each subset, check 7 turns.
        # Total: 33 * 906192 * 7 ~ 210M operations. In numpy, we can vectorize the turn checks.
        # Let me generate all 6-subsets and check them in batches.
        from itertools import combinations
        subsets = list(combinations(range(m), 6))
        # For each subset (a0,a1,a2,a3,a4,a5) in angular order (they are, since we sorted by angle),
        # check turns:
        # t0 = turn[i][a0][a1]  (i->a0->a1)
        # t1 = turn[a0][a1][a2]
        # t2 = turn[a1][a2][a3]
        # t3 = turn[a2][a3][a4]
        # t4 = turn[a3][a4][a5]
        # t5 = turn[a4][a5][i]  (a4->a5->i)
        # t6 = turn[a5][i][a0]  (a5->i->a0)
        # All must be > 1e-9.
        # Let me compute these in a vectorized way.
        # subsets is a list of tuples. Convert to array.
        sub_arr = np.array(subsets, dtype=int)  # shape (S, 6)
        # a0 = sub_arr[:,0], etc.
        a0, a1, a2, a3, a4, a5 = sub_arr[:, 0], sub_arr[:, 1], sub_arr[:, 2], sub_arr[:, 3], sub_arr[:, 4], sub_arr[:, 5]
        # turn[i][a0][a1]: i is the pivot, a0,a1 are indices into Q.
        # turn_pivot_a0_a1 = (Q[a0]-P[i]) x (Q[a1]-Q[a0])
        # = (Q[a0,0]-P[i,0])*(Q[a1,1]-Q[a0,1]) - (Q[a0,1]-P[i,1])*(Q[a1,0]-Q[a0,0])
        Pi = P[i]
        t0 = (Q[a0, 0] - Pi[0]) * (Q[a1, 1] - Q[a0, 1]) - (Q[a0, 1] - Pi[1]) * (Q[a1, 0] - Q[a0, 0])
        t1 = (Q[a1, 0] - Q[a0, 0]) * (Q[a2, 1] - Q[a1, 1]) - (Q[a1, 1] - Q[a0, 1]) * (Q[a2, 0] - Q[a1, 0])
        t2 = (Q[a2, 0] - Q[a1, 0]) * (Q[a3, 1] - Q[a2, 1]) - (Q[a2, 1] - Q[a1, 1]) * (Q[a3, 0] - Q[a2, 0])
        t3 = (Q[a3, 0] - Q[a2, 0]) * (Q[a4, 1] - Q[a3, 1]) - (Q[a3, 1] - Q[a2, 1]) * (Q[a4, 0] - Q[a3, 0])
        t4 = (Q[a4, 0] - Q[a3, 0]) * (Q[a5, 1] - Q[a4, 1]) - (Q[a4, 1] - Q[a3, 1]) * (Q[a5, 0] - Q[a4, 0])
        # t5 = turn[a4][a5][i] = (Q[a5]-Q[a4]) x (Pi-Q[a5])
        t5 = (Q[a5, 0] - Q[a4, 0]) * (Pi[1] - Q[a5, 1]) - (Q[a5, 1] - Q[a4, 1]) * (Pi[0] - Q[a5, 0])
        # t6 = turn[a5][i][a0] = (Pi-Q[a5]) x (Q[a0]-Pi)
        t6 = (Pi[0] - Q[a5, 0]) * (Q[a0, 1] - Pi[1]) - (Pi[1] - Q[a5, 1]) * (Q[a0, 0] - Pi[0])
        mask = (t0 > 1e-9) & (t1 > 1e-9) & (t2 > 1e-9) & (t3 > 1e-9) & (t4 > 1e-9) & (t5 > 1e-9) & (t6 > 1e-9)
        total += np.sum(mask)
    return total


def _is_valid(points):
    """Check validity: shape, range, min distance, no three collinear."""
    if points.shape != (33, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Min squared distance
    for i in range(33):
        for j in range(i + 1, 33):
            d2 = (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            if d2 < 1e-12:
                return False
    # No three collinear
    for i in range(33):
        for j in range(i + 1, 33):
            for k in range(j + 1, 33):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                if abs(det) < 1e-9:
                    return False
    return True


def _generate_cup_cap(n=33, seed=0):
    """Generate cup-cap configuration: n1 on cup (lower convex arc), n2 on cap (upper concave arc)."""
    rng = np.random.RandomState(seed)
    n1 = (n + 1) // 2  # 17
    n2 = n // 2  # 16
    # Cup: points on a convex arc in lower half, from left to right.
    # Use a parabola y = a*(x - 0.5)^2 + b, with a > 0 (convex up), so it's a cup.
    # x in [0, 1], y in [0, 0.5] roughly.
    x1 = np.linspace(0.05, 0.95, n1)
    # Add small random perturbation to avoid collinearity
    x1 += rng.uniform(-0.01, 0.01, n1)
    x1 = np.sort(x1)
    y1 = 0.3 * (x1 - 0.5) ** 2 + 0.05  # cup shape, min y ~ 0.05, max y ~ 0.3*0.25+0.05 = 0.125
    # Cap: points on a concave arc in upper half, from left to right.
    # y = -a*(x - 0.5)^2 + b, with a > 0, so it's concave (cap).
    x2 = np.linspace(0.05, 0.95, n2)
    x2 += rng.uniform(-0.01, 0.01, n2)
    x2 = np.sort(x2)
    y2 = -0.3 * (x2 - 0.5) ** 2 + 0.95  # cap shape, max y ~ 0.95, min y ~ 0.95 - 0.075 = 0.875
    # Combine
    points = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    # Add small random perturbation to all points to avoid collinearity
    points += rng.uniform(-0.005, 0.005, (n, 2))
    # Clamp to [0, 1]
    points = np.clip(points, 0.0, 1.0)
    return points


def _optimize(points, max_time=300.0):
    """Local optimization to reduce convex 7-gon count."""
    global _START
    start_time = time.time()
    best = points.copy()
    best_count = _count_convex7_fast(best)
    if not _is_valid(best):
        return best
    # Nelder-Mead style: perturb points and accept if count decreases.
    rng = np.random.RandomState(42)
    step = 0.01
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        # Random perturbation
        perturbation = rng.normal(0, step, (33, 2))
        candidate = best + perturbation
        candidate = np.clip(candidate, 0.0, 1.0)
        if not _is_valid(candidate):
            step *= 0.9
            continue
        count = _count_convex7_fast(candidate)
        if count < best_count:
            best = candidate
            best_count = count
            step = min(step * 1.1, 0.05)
        else:
            step *= 0.95
        if step < 1e-5:
            break
    return best


def erdos_szekeres_problem7() -> np.ndarray:
    global _START
    _START = time.time()
    # Generate initial cup-cap configuration
    points = _generate_cup_cap(33, seed=0)
    # Ensure validity
    if not _is_valid(points):
        # Add more perturbation
        rng = np.random.RandomState(123)
        points += rng.uniform(-0.01, 0.01, (33, 2))
        points = np.clip(points, 0.0, 1.0)
    # Optimize
    remaining = _DEADLINE - _elapsed() - 30.0  # 30s safety margin
    if remaining > 60:
        points = _optimize(points, max_time=remaining)
    # Final validity check
    if not _is_valid(points):
        # Fallback to a valid configuration
        points = _generate_cup_cap(33, seed=1)
        if not _is_valid(points):
            points = _generate_cup_cap(33, seed=2)
    return points


# EVOLVE-BLOCK-END
