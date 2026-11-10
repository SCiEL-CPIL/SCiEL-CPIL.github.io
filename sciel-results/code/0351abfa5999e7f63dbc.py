# EVOLVE-BLOCK-START
import numpy as np
import random
import math


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_7[alpha], alpha^2 = 3).

    The Nikodym property requires that for every point x in F_49^2, there is a
    direction v = (1, m) or v = (0, 1) such that all other points on the line
    through x in direction v belong to N.

    We aim to maximize the complement S = F_49^2 \ N (the excluded set).
    This is formulated as a Maximum Independent Set problem on a conflict graph,
    solved via multi-start greedy construction with randomized ordering.

    Returns:
        np.ndarray of shape (k, 2, 2) with dtype int64, listing points of N.
    """
    p = 7
    k = 3  # alpha^2 = 3 in F_7

    # Total number of points
    n_total = p * p * p * p  # 2401

    # Precompute all points as flat indices 0..2400
    # Index = ((xa * p + xb) * p + ya) * p + yb
    def to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Precompute all lines through each point
    # For each point, store list of (line_id, other_points)
    # A line through point i with direction (1, m) or (0, 1)
    # We represent lines by (start_point, direction) to avoid duplicates
    # But for our purposes, we just need: for each point, which lines pass through it,
    # and what are the other points on each line.

    # Directions: (1, m) for m in F_49, plus (0, 1)
    # F_49 elements: (a, b) where a, b in 0..6
    def add_field(a1, b1, a2, b2):
        return ((a1 + a2) % p, (b1 + b2) % p)

    def mul_field(a1, b1, a2, b2):
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + b1*a2)*alpha + b1*b2*alpha^2
        # = a1*a2 + b1*b2*k + (a1*b2 + b1*a2)*alpha
        ra = (a1 * a2 + b1 * b2 * k) % p
        rb = (a1 * b2 + b1 * a2) % p
        return ra, rb

    # All directions: (dx_a, dx_b) where dx != (0, 0)
    # We normalize directions so each line is counted once
    # Use canonical form: for direction (dx_a, dx_b), if dx_a != 0, we can scale to make dx_a = 1
    # But simpler: just enumerate all non-zero directions and deduplicate lines by their point set

    # Actually, for each point and each non-zero direction, the line is uniquely determined.
    # Two (point, direction) pairs define the same line if they have the same set of points.
    # For our algorithm, we need: for each point i, a list of "line groups" - each group is
    # the set of other points on a line through i in some direction.

    # Precompute: for each point, for each of the 49+1=50 directions, get the line points
    # But we need unique lines. A line through point i with direction d has 49 points.
    # The other 48 points are i + t*d for t in F_49 \ {0}.

    # Let's precompute for each point index, a list of tuples (direction_idx, set_of_other_points)
    # where direction_idx identifies the direction type

    # Directions: index 0 for (0, 1), indices 1..49 for (1, m) where m ranges over F_49
    # For (1, m): step is (1, m)
    # For (0, 1): step is (0, 1)

    # Precompute all 50 directions as (dx_a, dx_b)
    directions = []
    directions.append((0, 1))  # vertical: (0, 1)
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma * 0 + mb * 0))  # placeholder
    # Fix: for (1, m) where m = (ma, mb), the direction vector is (1, m) meaning
    # step adds 1 to x and m to y. So dx = (1, 0) in x-coords, dy = (ma, mb)
    # Wait, the problem says v = (1, m) for m in F_49. So x-coordinate step is 1 (the element 1 in F_49,
    # which is (1, 0)), and y-coordinate step is m.
    # So direction: x_step = (1, 0), y_step = m = (ma, mb)
    directions = []
    directions.append(((0, 0), (0, 1)))  # vertical: x_step=(0,0), y_step=(0,1)
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # x_step=(1,0), y_step=(ma,mb)

    n_dirs = len(directions)  # 50

    # Precompute lines: lines_through[i] = list of frozenset(other_points) for each direction
    # where i is the point index
    lines_through = [[] for _ in range(n_total)]
    for i in range(n_total):
        xa, xb, ya, yb = to_coords(i)
        for d_idx in range(n_dirs):
            (dx_a, dx_b), (dy_a, dy_b) = directions[d_idx]
            others = set()
            for t_a in range(p):
                for t_b in range(p):
                    if t_a == 0 and t_b == 0:
                        continue
                    # t = t_a + t_b*alpha
                    # x' = x + t * dx, y' = y + t * dy
                    # t * dx = t * (dx_a, dx_b)
                    tx_a, tx_b = mul_field(t_a, t_b, dx_a, dx_b)
                    ty_a, ty_b = mul_field(t_a, t_b, dy_a, dy_b)
                    nxa = (xa + tx_a) % p
                    nx_b = (xb + tx_b) % p
                    nya = (ya + ty_a) % p
                    ny_b = (yb + ty_b) % p
                    j = to_idx(nxa, nx_b, nya, ny_b)
                    others.add(j)
            lines_through[i].append(frozenset(others))

    # Build conflict graph adjacency
    # Two points conflict if they are on the same line (including vertical)
    # Build: conflicts[i] = set of j that conflict with i
    conflicts = [set() for _ in range(n_total)]
    for i in range(n_total):
        for line_pts in lines_through[i]:
            for j in line_pts:
                conflicts[i].add(j)

    # Multi-start greedy: build S (excluded set) as large as possible
    # S must be such that no two points in S are on the same line
    # i.e., S is an independent set in the conflict graph

    best_S = set()
    n_starts = 50
    rng = random.Random(42)

    for start in range(n_starts):
        S = set()
        order = list(range(n_total))
        rng.shuffle(order)
        for i in order:
            # Check if i can be added: no point in S conflicts with i
            can_add = True
            for j in conflicts[i]:
                if j in S:
                    can_add = False
                    break
            if can_add:
                S.add(i)
        if len(S) > len(best_S):
            best_S = S.copy()

    # Construct N = complement of S
    n_points = n_total - len(best_S)
    points = []
    for i in range(n_total):
        if i not in best_S:
            xa, xb, ya, yb = to_coords(i)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
