import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) = a + b*alpha
    def to_idx(x):
        return x[0] + x[1] * p

    def from_idx(idx):
        return (idx % p, idx // p)

    def field_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)

    def field_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)

    # Precompute all field elements as indices 0..48
    # idx = a + b*7
    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        xi = from_idx(i)
        for j in range(q):
            xj = from_idx(j)
            add_table[i, j] = to_idx(field_add(xi, xj))
            mul_table[i, j] = to_idx(field_mul(xi, xj))

    # A point in F_49^2 is (x, y) where x, y in F_49.
    # Point index = x_idx * q + y_idx, range 0..2400
    # Represent point as (x_idx, y_idx)

    # Directions: v = (1, m) for m in F_49 (49 directions) and v = (0, 1) (1 direction) = 50 directions
    # For direction (1, m): line through point (x0, y0): points (x0 + t, y0 + m*t) for t in F_49
    # For direction (0, 1): line through (x0, y0): points (x0, y0 + t) for t in F_49

    # Precompute for each point, the 50 lines through it, and for each line the 48 other points.
    # point_lines[point_idx] = list of 50 line_ids
    # line_points[line_id] = list of 49 point_idx (the points on the line)
    # line_id encoding: (direction, anchor) but better: a line is determined by direction + one point.
    # Total lines: 50 directions * 49 lines per direction (parallel classes) = 50 * 49 = 2450 lines.

    # Let's enumerate lines. For direction d (0..49), a line is identified by its "base" point in a canonical way.
    # For direction (1, m) [d = m, m in 0..48]: lines are parameterized by y-intercept. Line: y = m*x + c. 
    #   For a given c, the line contains points (x, m*x + c) for all x. There are 49 such lines (c in F_49).
    #   line_id for (m, c) = m * 49 + c  (range 0..48*49+48 = 2351)
    # For direction (0,1) [d = 49]: vertical lines x = c. 49 lines. line_id = 49*49 + c = 2401 + c (range 2401..2449)
    # Total line_ids: 0..2449 = 2450 lines.

    # For each line, store its 49 points.
    line_points = []  # line_points[line_id] = list of 49 point_idx
    # point_to_lines[point_idx] = list of 50 line_ids through that point

    # Build line_points
    # Direction (1, m), m = x_dir_idx in 0..48. m as field element = from_idx(m)
    for m_idx in range(q):
        m = from_idx(m_idx)
        for c_idx in range(q):
            c = from_idx(c_idx)
            line_id = m_idx * q + c_idx
            pts = []
            for x_idx in range(q):
                x = from_idx(x_idx)
                # y = m*x + c
                mx = field_mul(m, x)
                y = field_add(mx, c)
                y_idx = to_idx(y)
                point_idx = x_idx * q + y_idx
                pts.append(point_idx)
            line_points.append(pts)

    # Direction (0,1): vertical lines x = c
    for c_idx in range(q):
        line_id = q * q + c_idx  # 2401 + c_idx
        pts = []
        for y_idx in range(q):
            point_idx = c_idx * q + y_idx
            pts.append(point_idx)
        line_points.append(pts)

    # Now build point_to_lines: for each point, the 50 lines through it.
    point_to_lines = [None] * total_points
    for line_id in range(len(line_points)):
        for point_idx in line_points[line_id]:
            if point_to_lines[point_idx] is None:
                point_to_lines[point_idx] = []
            point_to_lines[point_idx].append(line_id)

    # Verify each point has exactly 50 lines
    # (skip verification for speed, trust construction)

    # clean_line_count[point_idx] = number of lines through point that contain no C-point
    # Initially all lines clean, so clean_line_count = 50 for all points.

    # For the greedy: when we add x to C, all 50 lines through x become non-clean.
    # For each such line L, the other 48 points on L lose L.
    # So for each of the 2400 other points z, decrement clean_line_count[z] by 1.

    # To check if adding x is valid: all 2400 affected z must have clean_line_count[z] >= 2 before decrement.
    # i.e., min over affected z of clean_line_count[z] >= 2.

    # Precompute for each point x, the list of the 2400 affected points (z != x on lines through x).
    # affected[x] = list of 2400 point_idx
    affected = []
    for x_idx in range(total_points):
        aff = []
        for line_id in point_to_lines[x_idx]:
            for z in line_points[line_id]:
                if z != x_idx:
                    aff.append(z)
        affected.append(aff)

    # Greedy: maintain C (set of excluded points), clean_line_count array.
    # Try to add points in a priority order.
    # Priority: use the Hermitian unital structure. Points in U (Tr(x)+N(y)=0) are good candidates.
    # Tr(x) = 2*x_a, N(y) = y_a^2 - 3*y_b^2
    def in_unital(point_idx):
        x_idx = point_idx // q
        y_idx = point_idx % q
        xa = x_idx % p
        ya = y_idx % p
        yb = y_idx // p
        return (2 * xa + ya * ya - 3 * yb * yb) % p == 0

    # Build candidate order: unital points first, then others.
    unital_points = [i for i in range(total_points) if in_unital(i)]
    other_points = [i for i in range(total_points) if not in_unital(i)]
    candidate_order = unital_points + other_points

    # Greedy with multiple passes / restarts
    best_C = set()

    def run_greedy(order, time_limit):
        C = set()
        clean_line_count = np.full(total_points, 50, dtype=np.int32)
        # For speed, track clean_line_count as a numpy array.
        # When adding x: check min(clean_line_count[affected[x]]) >= 2, then decrement.
        t_start = time.time()
        for x in order:
            if time.time() > time_limit:
                break
            if x in C:
                continue
            aff = affected[x]
            # Check if all affected have count >= 2
            # Use numpy for speed
            min_count = np.min(clean_line_count[aff])
            if min_count >= 2:
                C.add(x)
                clean_line_count[aff] -= 1
        return C

    # Run greedy with the priority order
    C = run_greedy(candidate_order, deadline - safety_margin)
    best_C = C

    # Try to improve with additional passes: re-scan and try to add more points
    # After the first pass, some points might now be addable (but counts only decrease, so no new points become addable).
    # Actually counts only decrease, so the set of addable points shrinks. No improvement from re-scan.
    # Instead, try different orderings for diversity.

    # Try a few different orderings within remaining time
    remaining = deadline - safety_margin - time.time()
    if remaining > 5:
        # Random shuffles of the non-unital part
        import random
        rng = random.Random(42)
        for trial in range(10):
            if time.time() > deadline - safety_margin:
                break
            order2 = unital_points + other_points[:]
            rng.shuffle(order2)
            C2 = run_greedy(order2, deadline - safety_margin)
            if len(C2) > len(best_C):
                best_C = C2

    # Build the Nikodym set N = all points not in best_C
    N_points = []
    for point_idx in range(total_points):
        if point_idx not in best_C:
            x_idx = point_idx // q
            y_idx = point_idx % q
            xa = x_idx % p
            xb = x_idx // p
            ya = y_idx % p
            yb = y_idx // p
            N_points.append([[xa, xb], [ya, yb]])

    if not N_points:
        # Fallback: return all points (valid Nikodym set)
        N_points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
