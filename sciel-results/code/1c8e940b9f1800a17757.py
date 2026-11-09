# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.

    Field: F_49 = F_7[alpha] with alpha^2 = 3. A point is [[xa, xb], [ya, yb]].

    The Nikodym condition: for every point x in F_49^2, there exists a direction
    v = (1, m) for some m in F_49, or v = (0, 1), such that all points on the
    line through x in direction v (other than x itself) belong to N.

    We use a constructive approach based on blocking sets. The idea:
    - For each slope m (including infinity), we need a blocking set B_m such that
      every line of slope m intersects B_m (i.e., for every point x on a line L
      of slope m, at least one other point of L is in N, which means at least
      one point of L \\ {x} is in N).

    A simpler sufficient condition: N is a Nikodym set if for every point x and
    every line L through x, L \\ {x} subset N. But we only need ONE direction per
    point. So we can assign to each point x a "witness direction" d(x) such that
    all other points on the line through x in direction d(x) are in N.

    Strategy: Use a greedy / assignment approach.
    - Start with N = all points (complement empty).
    - We want to remove as many points as possible from N while maintaining the
      Nikodym property.
    - A point p can be removed from N (i.e., added to complement) iff for every
      line L through p, there exists at least one direction d such that L_d \\ {p}
      is still fully in N. Actually, we need: for the specific direction d(p)
      assigned to p, L_{d(p)} \\ {p} subset N. But if p is not in N, we don't need
      a witness for p. We need witnesses for points IN N? No: "every point x of
      F_49^2 (in N or not) must lie on a line ... whose points other than x all
      belong to N."

    So EVERY point x (whether in N or not) needs a direction d such that
    {x + t*v : t in F_49, t != 0} subset N.

    This means: for each x, there must exist v such that all q = x + t*v (t != 0)
    are in N.

    If x is not in N, that's fine as long as the other points on some line through
    x are in N.

    Key insight: If we make N very large, it's easy. We want N small.

    Alternative: Think of the complement C = F_49^2 \\ N. We want C large.
    For each x in F_49^2, there must exist a direction v such that
    (line through x in direction v) \\ {x} subset N = F_49^2 \\ C.
    Equivalently: (line through x in direction v) \\ {x} intersects C in at most... 
    no, it must be DISJOINT from C. So (line through x in direction v) \\ {x} 
    must contain NO points of C.

    So: for each x, there exists v such that C ∩ (line_v(x) \\ {x}) = empty.

    If x ∈ C, then line_v(x) \\ {x} must be disjoint from C, meaning all other
    points on that line are in N.
    If x ∉ C (x ∈ N), same condition: line_v(x) \\ {x} disjoint from C.

    So the condition is: for every x, there is a direction v such that
    C ∩ (line_v(x) \\ {x}) = ∅.

    This is equivalent to: for every x, the set of directions v for which
    C ∩ (line_v(x) \\ {x}) ≠ ∅ does NOT cover all directions. I.e., there is at
    least one "clean" direction for x.

    Let's try a structured construction.

    Consider the following: partition F_49^2 into lines of a fixed slope, say
    slope 0 (horizontal lines). Each horizontal line has 49 points. If we put
    all points of a horizontal line into C, then for a point x on that line,
    the horizontal line through x has 48 other points all in C, so horizontal
    direction is NOT clean for x. But x might have another clean direction.

    Let's try: C = union of some horizontal lines. Say we take r horizontal lines
    as C. Then |C| = 49r. For a point x on one of these C-lines, we need some
    non-horizontal direction v such that line_v(x) \\ {x} has no C-points.
    For a point x NOT on a C-line, any direction works as long as line_v(x) \\ {x}
    has no C-points. Since x is not on a C-line, if we choose v to be horizontal,
    line_0(x) is the horizontal line through x, which has no C-points (since x is
    not on a C-line and the C-lines are specific horizontal lines). Wait, the
    horizontal line through x is the set of all points with the same y-coordinate
    as x. If x's y-coordinate is not one of the C-line y-coordinates, then the
    entire horizontal line through x is disjoint from C. So for x not on a C-line,
    horizontal direction is clean. Good.

    For x on a C-line (y = y0, where y0 is a C-line), we need a non-horizontal
    direction. Consider direction v = (1, m). The line through x = (x0, y0) in
    direction (1, m) is { (x0 + t, y0 + m*t) : t in F_49 }. The other points are
    (x0 + t, y0 + m*t) for t != 0. We need none of these to be in C. C consists
    of points with y-coordinate in {y0, y1, ..., y_{r-1}}. So we need: for all
    t != 0, y0 + m*t is not in {y0, y1, ..., y_{r-1}}.
    y0 + m*t = y0 implies m*t = 0, so t = 0 (since m != 0) or m = 0. If m != 0,
    then t = 0 is the only solution, so y0 + m*t != y0 for t != 0. Good.
    But we also need y0 + m*t != yj for j != 0. So we need m*t != yj - y0 for
    all t != 0 and all j != 0. This means m*t avoids the set {yj - y0 : j = 1..r-1}
    for all t != 0. The set {m*t : t != 0} = F_49 \\ {0}. So we need
    F_49 \\ {0} ∩ {yj - y0 : j = 1..r-1} = empty, i.e., yj - y0 = 0 for all j,
    which is false. So this doesn't work for r >= 2.

    So taking entire horizontal lines as C doesn't work for r >= 2.

    Let's try a different approach. Use a known construction: the "Nikodym set"
    from finite geometry. For F_q^2 with q = p^2, there are constructions of
    size q^2 - q^{3/2} or similar.

    Actually, let me think about the minimum size. The complement C must satisfy:
    for each x, there is a direction v with C ∩ (line_v(x) \\ {x}) = ∅.

    This is equivalent to: the family of sets {line_v(x) \\ {x} : v} for each x
    must not all intersect C. I.e., C is not a "blocking set" for the hypergraph
    where hyperedges are line_v(x) \\ {x}.

    Let's try a probabilistic / random approach with local improvement, but with
    a time limit.

    Actually, let me try a simpler structured approach.

    Consider C as a "graph" of a function: C = { (x, f(x)) : x in F_49 } for
    some function f: F_49 -> F_49. Then |C| = 49. For a point (x, y) with
    y = f(x) (in C), we need a direction v such that line_v((x,y)) \\ {(x,y)}
    has no C-points. For a point (x, y) with y != f(x) (in N), same.

    For direction v = (1, m): line through (x, y) is { (x+t, y+m*t) : t }.
    C-points on this line: (x+t, f(x+t)) = (x+t, y+m*t) => f(x+t) = y + m*t.
    We need: for the chosen m, for all t != 0, f(x+t) != y + m*t.

    This is getting complex. Let me just implement a greedy algorithm with a
    time limit.

    Greedy: Start with C = empty, N = all. Try to add points to C one by one.
    When adding point p to C, we must check that for every point x in F_49^2,
    there still exists a direction v such that C ∩ (line_v(x) \\ {x}) = ∅.
    This is expensive to check.

    Alternative: Start with N = all, try to remove points from N (add to C).
    A point p can be removed if for every x, the condition still holds. But the
    condition for x depends on C.

    Let me try a different tack. Use the following known result: In F_q^2, a
    set N is a Nikodym set if for every point x, there is a line through x
    whose other points are in N. The minimum size is at least q^2 - q^{3/2} + ... 
    but I'm not sure.

    Let me just do a simple construction: N = all points except those on a
    specific set of lines. 

    Actually, let me try: C = { (x, y) : y = x^2 } (a parabola). |C| = 49.
    For a point (x, x^2) in C: we need a direction v such that line_v((x,x^2))
    \\ {(x,x^2)} has no parabola points.
    For direction (1, m): points (x+t, x^2 + m*t). We need x^2 + m*t != (x+t)^2
    for all t != 0. (x+t)^2 = x^2 + 2xt + t^2. So x^2 + m*t = x^2 + 2xt + t^2
    => m*t = 2xt + t^2 => t(m - 2x - t) = 0. For t != 0, m - 2x - t = 0 => t = m - 2x.
    So there's exactly one t = m - 2x (if m - 2x != 0) that gives a parabola point.
    So for most m, there is exactly one other parabola point on the line. We need
    a direction with ZERO other parabola points.
    For direction (0, 1): vertical line x = const. Points (x, x^2 + t). 
    (x, x^2 + t) is on parabola iff x^2 + t = x^2 => t = 0. So no other parabola
    points! So vertical direction works for all parabola points.
    For a point (x, y) not on parabola (y != x^2): vertical line x = const has
    points (x, y + t). (x, y+t) on parabola iff y + t = x^2 => t = x^2 - y.
    So there's exactly one t giving a parabola point (unless x^2 - y = 0, but
    y != x^2 so t != 0). So vertical direction has one parabola point other than
    possibly the point itself. Wait, the point (x, y) is not on parabola, so
    (x, y+t) = (x, x^2) when t = x^2 - y != 0. So there IS a parabola point on
    the vertical line. So vertical direction does NOT work for non-parabola points.
    We need another direction for non-parabola points.

    This is getting complicated. Let me just use a time-limited greedy search.

    Given the 600s limit, let me implement a simple approach:
    1. Start with C = empty.
    2. For each point in random order, try to add it to C.
    3. To check if adding p to C is valid: for every point x, check if there
       exists a direction v such that (C ∪ {p}) ∩ (line_v(x) \\ {x}) = ∅.
       This is O(49^2 * 50 * 49) per addition, which is too slow.

    Let me think differently. The condition for C is:
    For every x, the number of directions v with C ∩ (line_v(x) \\ {x}) != ∅
    is less than 50 (total directions).

    When we add p to C, for each x != p, the new C may block some directions
    for x. Specifically, for x != p, direction v is blocked for x iff p is on
    line_v(x) \\ {x}, i.e., p = x + t*v for some t != 0.

    So adding p blocks, for each x != p, the directions v such that p - x is
    a nonzero multiple of v. For a given x != p, p - x is a fixed nonzero vector.
    The directions v that are parallel to p - x: v = (1, m) where m = (p_y - x_y)
    / (p_x - x_x) if p_x != x_x, or v = (0, 1) if p_x = x_x. So for each x != p,
    exactly ONE direction is newly blocked (the direction from x to p).

    So when we add p, for each x != p, one direction becomes blocked (if it wasn't
    already). For x = p, all 50 directions are blocked (since p is in C, and
    line_v(p) \\ {p} contains... wait, p is in C, so for x = p, we need a direction
    v such that C ∩ (line_v(p) \\ {p}) = ∅. But p is in C, and line_v(p) \\ {p}
    does not contain p, so p being in C doesn't directly block directions for x = p.
    However, other points in C on line_v(p) \\ {p} do block.

    OK so the incremental check: when adding p to C, for each x != p, we add one
    newly blocked direction (the direction from x to p). For x = p, we need to
    check if there's still a clean direction.

    We can maintain for each x the count of blocked directions. Initially 0.
    When adding p, for each x != p, increment blocked_count[x] by 1 (for the
    direction from x to p). For x = p, we need to check if blocked_count[p] < 50
    (i.e., there's at least one clean direction). But blocked_count[p] was already
    tracking directions blocked by existing C-points. Adding p itself doesn't
    block a direction for x = p (since p is not on line_v(p) \\ {p}). So
    blocked_count[p] doesn't change when we add p.

    Wait, but we also need to make sure that for x = p, after adding p, there's
    still a clean direction. Since p is in C, and line_v(p) \\ {p} doesn't contain
    p, the only thing that matters is whether line_v(p) \\ {p} contains other C-points.
    This is already captured by blocked_count[p] (which counts directions v such
    that C ∩ (line_v(p) \\ {p}) != ∅). Adding p doesn't change this.

    So the algorithm:
    - blocked_count[x] = number of directions v such that C ∩ (line_v(x) \\ {x}) != ∅.
    - Initially C = empty, blocked_count[x] = 0 for all x.
    - To add p: 
      - For each x != p, determine the direction v_x from x to p. Increment
        blocked_count[x] by 1 ONLY IF this direction was not already blocked for x.
        But we don't track WHICH directions are blocked, just the count. So we
        might double-count.

    We need to track the set of blocked directions for each x, or at least know
    if a specific direction is already blocked.

    This is O(49^2) memory and O(49) per addition (to update one direction for
    each x). Total O(49^3) = ~117,000 operations, which is fine.

    Let me implement this.

    Directions: 49 slopes m in F_49, plus vertical. Total 50 directions.
    For each x, maintain a set of blocked direction indices (0..49).

    When adding p, for each x != p:
      - Compute direction from x to p.
      - If this direction is not already in blocked_dirs[x], add it and increment
        blocked_count[x].
      - If blocked_count[x] becomes 50, then x has no clean direction -> invalid.

    For x = p: blocked_dirs[p] is unchanged. Check if blocked_count[p] < 50.

    If valid, add p to C.

    We try to add points in some order (random or structured) to maximize |C|.

    Let me code this up with a time limit.

    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    start_time = time.time()
    deadline = start_time + 550  # 550s safety margin

    # Precompute directions. Direction index 0..48: slope m = index (as field element).
    # Direction index 49: vertical (0, 1).
    # Field elements are pairs (a, b) for a + b*alpha, alpha^2 = 3.
    # But for direction (1, m), m is a field element. For vertical, v = (0, 1).

    # Represent field elements as integers 0..48, where element i = (a, b) with
    # i = a + b*7, a = i % 7, b = i // 7.
    # alpha^2 = 3.
    # Addition: (a1, b1) + (a2, b2) = ((a1+a2)%7, (b1+b2)%7).
    # Multiplication: (a1, b1)*(a2, b2) = (a1*a2 + b1*b2*3, a1*b2 + b1*a2) mod 7.

    def field_add(x, y):
        xa, xb = x % 7, x // 7
        ya, yb = y % 7, y // 7
        return ((xa + ya) % 7) + ((xb + yb) % 7) * 7

    def field_mul(x, y):
        xa, xb = x % 7, x // 7
        ya, yb = y % 7, y // 7
        ra = (xa * ya + xb * yb * 3) % 7
        rb = (xa * yb + xb * ya) % 7
        return ra + rb * 7

    def field_sub(x, y):
        return field_add(x, p - y if y != 0 else 0)

    def field_neg(x):
        return p - x if x != 0 else 0

    # Precompute field tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = field_add(i, j)
            mul_table[i, j] = field_mul(i, j)

    # Point (x, y) where x, y are field elements (0..48).
    # Point index: x * q + y, so 0..2400.
    # x = point_idx // q, y = point_idx % q.

    # For each pair of distinct points, compute the direction from point1 to point2.
    # Direction is an index 0..49.
    # If x2 != x1: slope m = (y2 - y1) / (x2 - x1). Direction index = m (0..48).
    # If x2 == x1: vertical, direction index = 49.

    # Precompute direction from point i to point j (i != j).
    # dir_table[i, j] = direction index.
    dir_table = np.zeros((total_points, total_points), dtype=np.int64)
    for i in range(total_points):
        xi = i // q
        yi = i % q
        for j in range(total_points):
            if i == j:
                dir_table[i, j] = -1
                continue
            xj = j // q
            yj = j % q
            dx = field_sub(xj, xi)
            dy = field_sub(yj, yi)
            if dx == 0:
                dir_table[i, j] = 49
            else:
                # m = dy / dx. Need inverse of dx.
                # Find inverse: dx * inv = 1.
                inv = None
                for t in range(q):
                    if field_mul(dx, t) == 1:
                        inv = t
                        break
                m = field_mul(dy, inv)
                dir_table[i, j] = m

    # Now greedy: try to add points to C.
    # blocked_dirs[x] = set of blocked direction indices for point x.
    # We use a list of sets, or a 2D boolean array.
    blocked = np.zeros((total_points, 50), dtype=np.bool_)
    blocked_count = np.zeros(total_points, dtype=np.int64)

    C = set()

    # Try points in random order. Use a fixed seed for reproducibility.
    np.random.seed(42)
    order = np.random.permutation(total_points)

    for idx in order:
        if time.time() > deadline:
            break

        p_idx = int(idx)
        # Check if adding p_idx is valid.
        # For x = p_idx: check blocked_count[p_idx] < 50.
        if blocked_count[p_idx] >= 50:
            continue

        # For x != p_idx: check if the new direction would make blocked_count[x] >= 50.
        # We need to check for each x != p_idx if dir_table[x, p_idx] is not already
        # blocked for x. If it is already blocked, no change. If not, it would increment
        # blocked_count[x] by 1. If that makes it 50, invalid.

        valid = True
        # Check all x != p_idx
        for x in range(total_points):
            if x == p_idx:
                continue
            d = dir_table[x, p_idx]
            if not blocked[x, d]:
                # This direction would be newly blocked for x
                if blocked_count[x] + 1 >= 50:
                    valid = False
                    break

        if valid:
            # Add p_idx to C
            C.add(p_idx)
            # Update blocked for x != p_idx
            for x in range(total_points):
                if x == p_idx:
                    continue
                d = dir_table[x, p_idx]
                if not blocked[x, d]:
                    blocked[x, d] = True
                    blocked_count[x] += 1

    # Build N = all points not in C.
    n_points = []
    for i in range(total_points):
        if i not in C:
            x = i // q
            y = i % q
            xa, xb = x % 7, x // 7
            ya, yb = y % 7, y // 7
            n_points.append([[xa, xb], [ya, yb]])

    return np.array(n_points, dtype=np.int64)


# EVOLVE-BLOCK-END
