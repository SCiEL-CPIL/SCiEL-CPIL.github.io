# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy local-search /
    Max-SAT-style heuristic.

    Field: F_49 = F_7[alpha] with alpha^2 = 3 (3 is a quadratic non-residue mod 7).
    A point is (x, y) where x, y in F_49, written as [[xa, xb], [ya, yb]].

    Nikodym condition: for every point x in F_49^2, there exists a direction
    v (either (1, m) for some m in F_49, or (0, 1)) such that every other
    point on the line x + t*v (t != 0) belongs to N.

    We build the complement S = F_49^2 \\ N. The condition is:
      For every point p and every direction v, the line L(p, v) contains
      at least one point of S other than p itself.
      (Because if NO point of L(p,v) \\ {p} is in S, then ALL of them are in N,
       which is fine... wait, let me re-read.)

    Actually: N must contain all of L(p,v) \\ {p} for SOME v.
    Equivalently: for each p, there exists v such that L(p,v) \\ {p} ⊆ N.
    In terms of S: for each p, there exists v such that L(p,v) \\ {p} ∩ S = ∅.
    i.e., there exists a direction v for p such that NO other point on that
    line through p is in S.

    So S must be a set such that for every point p, there is at least one
    direction v where the entire line (minus p) avoids S.

    Greedy approach: start with S = empty (N = everything, trivially valid).
    Try to add points to S one by one. A point p can be added to S if, after
    adding, for every point q, there still exists a direction v_q such that
    L(q, v_q) \\ {q} ∩ S_new = ∅.

    Adding p to S "breaks" a direction v for point q if p lies on L(q, v)
    and p != q. So p can be added iff for every q, not ALL of q's directions
    are broken by S_new.

    We use a heuristic: maintain for each point q the set of "safe" directions
    (directions v where L(q,v) \\ {q} ∩ S = ∅). A point p can be added iff
    for every q, the number of safe directions for q remains > 0 after adding p.

    For efficiency, we precompute which points lie on which (point, direction)
    lines. Then adding p to S removes p from the "safe" status of direction v
    for point q iff p is on L(q,v), p != q, and v was previously safe for q
    (i.e., no other S point was on L(q,v) \\ {q}).

    Actually simpler: maintain a count safe_count[q] = number of directions v
    such that L(q,v) \\ {q} ∩ S = ∅. Initially all 49 directions are safe
    for each q (49 slopes + vertical = 50 directions? Let's check: v = (1,m)
    for m in F_49 gives 49 directions, plus v = (0,1) gives 1 more, total 50).

    When we add p to S: for each q != p, for each direction v, if p is on
    L(q,v), then if this was the FIRST S point on L(q,v) \\ {q}, safe_count[q]
    decreases by 1. We need to track for each (q, v) whether any S point is
    on L(q,v) \\ {q}.

    This is expensive. Let's use a different approach: for each (q, v), track
    the number of S points on L(q,v) \\ {q}. safe_count[q] = number of v where
    this count is 0.

    Total (q, v) pairs: 2401 * 50 = 120050. Each line has 49 points (other than
    q). When we add p to S, we need to find all (q, v) such that p is on
    L(q, v) and q != p. For a fixed p, how many such (q, v) pairs? For each
    direction v, the line through p in direction v contains 49 other points q.
    So 50 * 49 = 2450 pairs. For each, we increment the count. If the count
    goes from 0 to 1, safe_count[q] decreases.

    This is O(2450) per addition, which is feasible.

    We greedily add points to S in an order that minimizes the damage.
    """
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    deadline = time.time() + 590  # safety margin

    # Precompute line membership: for each point index and direction,
    # which other points are on the line.
    # Directions: 0..48 are slopes m (direction (1, m)), direction 49 is vertical (0, 1).
    # Point index: idx = ((xa * p + xb) * p + ya) * p + yb  -> 0..2400
    # Actually let's use: idx = xa * q * q + xb * q + ya * p + yb
    # where xa, xb, ya, yb in 0..6

    def to_idx(xa, xb, ya, yb):
        return xa * q * q + xb * q + ya * p + yb

    def from_idx(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    total_points = q * q  # 2401
    num_dirs = q + 1  # 50

    # For each direction, compute the line through each point.
    # line_points[dir][point_idx] = list of other point indices on the line
    # This is 50 * 2401 * 49 entries = ~5.9M, might be heavy but let's try.
    # Alternatively, compute on the fly.

    # Let's precompute for each direction the equivalence classes (lines).
    # For direction d:
    #   if d < 49: slope m = d, line through (x, y) has points (x + t, y + m*t) for t in F_49
    #   if d == 49: vertical, line through (x, y) has points (x, y + t) for t in F_49

    # Precompute: for each direction d and each point, the set of other points on the line.
    # We'll store as: line_members[d][idx] = list of 49 indices (excluding idx itself)

    # Actually, for the greedy algorithm, when adding point p, we need to find
    # all (q, d) such that p is on line(q, d). This is equivalent to: for each
    # direction d, find all q such that p is on line(q, d). For a fixed d and p,
    # the line through p in direction d contains 50 points (including p). So
    # there are 49 other points q on that line.

    # Let's precompute: for each direction d and point p, the list of other
    # points on the line through p in direction d.
    # line_through[d][p] = list of 49 indices

    line_through = [[None] * total_points for _ in range(num_dirs)]

    # Field operations
    def add_f(a, b):  # a, b are (ra, rb)
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    # Precompute lines
    for d in range(num_dirs):
        for idx in range(total_points):
            xa, xb, ya, yb = from_idx(idx)
            x = (xa, xb)
            y = (ya, yb)
            members = []
            for t in range(q):
                if t == 0:
                    continue
                if d < q:
                    # direction (1, m) where m = d
                    m = (1, d % q)  # Wait, m is in F_49, so m = (m_a, m_b)
                    # Actually d is 0..48, representing m in F_49
                    # m = (d // p, d % p) ? No, d is just an index.
                    # Let me re-index: direction d (0..48) corresponds to m = (d // p, d % p)
                    m_a = d // p
                    m_b = d % p
                    m = (m_a, m_b)
                    # new_x = x + t * 1 = x + t
                    # new_y = y + t * m
                    t_f = (t, 0)
                    t_one = add_f((0, 0), t_f)  # just t as field element
                    # t is 0..48, representing field element (t//p, t%p)
                    t_f = (t // p, t % p)
                    new_x = add_f(x, t_f)
                    tm = mul_f(t_f, m)
                    new_y = add_f(y, tm)
                    nidx = to_idx(new_x[0], new_x[1], new_y[0], new_y[1])
                    members.append(nidx)
                else:
                    # vertical direction (0, 1)
                    t_f = (t // p, t % p)
                    new_x = x
                    new_y = add_f(y, t_f)
                    nidx = to_idx(new_x[0], new_x[1], new_y[0], new_y[1])
                    members.append(nidx)
            line_through[d][idx] = members

    # Initialize
    S = [False] * total_points  # S[i] = True if point i is in complement
    # count_s_on_line[d][idx] = number of S points on line_through[d][idx]
    count_s_on_line = [[0] * total_points for _ in range(num_dirs)]
    # safe_count[idx] = number of directions d where count_s_on_line[d][idx] == 0
    safe_count = [num_dirs] * total_points

    # Greedy: repeatedly add the point that causes the least damage
    # Damage of adding p: for each d, for each q in line_through[d][p] (q != p),
    # if count_s_on_line[d][q] was 0, it becomes 1, so safe_count[q] decreases.
    # We want to minimize the number of q whose safe_count drops to 0.

    # For efficiency, maintain a priority or just scan.
    # 2401 points, each addition costs O(50 * 49) = 2450 for the update.
    # Finding the best point to add: scanning all 2401 points and computing
    # damage for each is O(2401 * 2450) = ~5.9M per iteration. Too slow.

    # Alternative: just add points in a random or structured order, checking
    # validity incrementally. Or use a simpler heuristic.

    # Let's try: add points in a structured order (e.g., by y-coordinate
    # non-special first, matching the baseline), and greedily skip if invalid.

    # Actually, let's just do a simple greedy: iterate through points in a
    # shuffled order, try to add each, accept if valid.

    order = list(range(total_points))
    random.seed(42)
    random.shuffle(order)

    for idx in order:
        if time.time() > deadline:
            break
        if S[idx]:
            continue
        # Check if adding idx is valid
        # Temporarily add and check
        valid = True
        changed = []  # (d, q, old_count) for rollback
        for d in range(num_dirs):
            for q_idx in line_through[d][idx]:
                if q_idx == idx:
                    continue
                old = count_s_on_line[d][q_idx]
                if old == 0:
                    # safe_count[q_idx] would decrease
                    if safe_count[q_idx] == 1:
                        valid = False
                        break
                changed.append((d, q_idx, old))
            if not valid:
                break
        if not valid:
            continue
        # Commit
        S[idx] = True
        for d, q_idx, old in changed:
            count_s_on_line[d][q_idx] = old + 1
            if old == 0:
                safe_count[q_idx] -= 1

    # Build result: N = all points not in S
    points = []
    for idx in range(total_points):
        if not S[idx]:
            xa, xb, ya, yb = from_idx(idx)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
