import numpy as np
from itertools import product

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]

    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Elements represented as tuples (a, b) meaning a + b*alpha

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)

    def inv(a):
        a0, a1 = a
        norm = (a0 * a0 + a1 * a1 * k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p - 2, p)
        a_p = (a0, (-a1) % p)
        return mul(a_p, (norm_inv, 0))

    # Generate all field elements
    F_q = [(a, b) for a in range(p) for b in range(p)]
    zero = (0, 0)
    one = (1, 0)

    # Generate all points in F_49^2
    all_points = []
    for x in F_q:
        for y in F_q:
            all_points.append((x, y))

    point_to_idx = {pt: i for i, pt in enumerate(all_points)}

    # Precompute all lines in F_49^2
    # A line is determined by a point and a direction.
    # Directions: v = (1, m) for m in F_q, and v = (0, 1) for vertical lines
    # Total directions: q + 1 = 50

    # For each direction, and each "offset", generate the line
    # Actually, easier: for each point and each direction, generate the line through that point
    # But that's redundant. Let's just enumerate all lines.

    # A line can be represented by its set of points.
    # Number of lines in F_q^2: q(q+1)(q^2+1)/(q+1) ... no.
    # Number of lines = q^2(q+1) ... let me just generate them.

    # For each direction d (q+1 of them), and for each "base" point,
    # the line is {base + t*d : t in F_q}
    # But different base points give the same line.
    # To avoid duplicates, we can fix a canonical representative.

    # Simpler approach: for each point and each direction, compute the line,
    # then deduplicate.

    # Actually, let's just compute for each point, which lines pass through it.
    # For the Nikodym property, we need: for every point x, there exists a line L through x
    # such that L \ {x} is disjoint from C (the complement).

    # Let's precompute: for each point, the list of lines through it.
    # Each line is a frozenset of point indices.

    # Directions
    directions = []
    for m in F_q:
        directions.append(((1, 0), m))  # (1, m)
    directions.append(((0, 1), zero))   # (0, 1) vertical

    # For each point and each direction, compute the line
    # line through point p with direction d: {p + t*d : t in F_q}

    # Let's precompute lines as sets of point indices
    # Key: (point_index, direction_index) -> line (set of point indices)
    # But we want to deduplicate.

    # Alternative: just for each point, store the lines through it.
    # lines_through_point[i] = list of frozensets of point indices

    lines_through_point = [[] for _ in range(q * q)]

    for d_idx, d in enumerate(directions):
        dx, dy = d
        for p_idx, p in enumerate(all_points):
            # Generate line: p + t*d for t in F_q
            line_points = set()
            for t in F_q:
                # t * d
                tx, ty = t
                # t*dx, t*dy
                tdx = mul(t, dx)
                tdy = mul(t, dy)
                # p + (tdx, tdy)
                new_point = add(p, (tdx, tdy))
                line_points.add(point_to_idx[new_point])
            line_frozen = frozenset(line_points)
            lines_through_point[p_idx].append(line_frozen)

    # Now, we want to find a large set C such that for every point x in C,
    # there exists a line L through x with L \ {x} disjoint from C.
    # Equivalently, for every x in C, there exists a line L through x such that
    # L ∩ C = {x}.

    # Let's try to construct C greedily or using a known structure.

    # Known: the complement of a conic works.
    # A non-degenerate conic in F_q^2 has q+1 points if it doesn't pass through infinity,
    # or q points if it does.

    # Let's try C = { (x, y) : y = x^2 } — this has q = 49 points.
    # Check: for each point (x, x^2) on the parabola, is there a line through it
    # that meets the parabola only at that point?
    # The tangent line at (x, x^2) meets the parabola only at that point (with multiplicity 2,
    # but in finite fields, the tangent intersects only at that point if the field is small enough?
    # Actually, in F_q, the tangent to y=x^2 at x0 is y = 2x0*x - x0^2.
    # Intersection with y=x^2: x^2 = 2x0*x - x0^2 => (x-x0)^2 = 0 => x = x0.
    # So yes, the tangent meets the parabola only at (x0, x0^2).
    # So C = parabola is a valid Nikodym complement.

    # But can we do better? A conic with q+1 = 50 points.
    # Let's try x^2 + y^2 = 1. How many points?
    # In F_7, count solutions to x^2 + y^2 = 1.
    # Squares in F_7: 0, 1, 2, 4.
    # x^2 + y^2 = 1:
    # (0,1),(0,6),(1,0),(6,0),(2,3),(2,4),(3,2),(4,2),(3,4),(4,3)
    # Wait, let me count properly.
    # x^2 can be 0,1,2,4. y^2 = 1 - x^2.
    # x^2=0: y^2=1: y=1,6. 2 points.
    # x^2=1: y^2=0: y=0. x=1,6. 2 points.
    # x^2=2: y^2=6: 6 not a square. 0.
    # x^2=4: y^2=4: y=2,5. x=2,5. 4 points.
    # Total: 2+2+0+4 = 8 points in F_7.
    # In F_49, it should have more. A non-degenerate conic over F_q has q+1 points.
    # So x^2 + y^2 = 1 should have 50 points if it's non-degenerate.

    # Let's just use the parabola for simplicity and correctness, then try to improve.

    # Actually, let's try to find the maximum C using a greedy approach.
    # Start with C = empty, try to add points one by one.
    # A point x can be added to C if for every line L through x, L ∩ C != {x} is not required...
    # Wait, the condition is: for every x in C, there EXISTS a line L through x with L ∩ C = {x}.
    # So when adding x to C, we need to check that there's still a line through x
    # that doesn't contain any other point of C.

    # Greedy: start with C = all points, then remove points that violate the condition?
    # No, that's hard.

    # Let's just use the parabola: C = {(x, x^2) : x in F_49}
    C_set = set()
    for x in F_q:
        y = mul(x, x)
        C_set.add((x, y))

    # Verify validity
    def is_valid(C_set):
        for x in C_set:
            x_idx = point_to_idx[x]
            found = False
            for line in lines_through_point[x_idx]:
                # Check if line ∩ C = {x}
                intersection = line & C_set
                if intersection == {x_idx}:
                    found = True
                    break
            if not found:
                return False
        return True

    # The parabola should be valid. Let's verify.
    # But let's also try to find a larger C.

    # Try: C = conic x^2 + y^2 + xy + x + y = 0 or something.
    # For now, let's use the parabola and see if we can improve.

    # Actually, let's try a different approach: start with a large candidate set
    # and iteratively remove points that don't satisfy the condition.

    # Start with C = all points
    C_candidate = set(range(q * q))

    # Iteratively remove points that don't have a "free" line
    changed = True
    while changed:
        changed = False
        for x_idx in list(C_candidate):
            has_free_line = False
            for line in lines_through_point[x_idx]:
                if line & C_candidate == {x_idx}:
                    has_free_line = True
                    break
            if not has_free_line:
                C_candidate.discard(x_idx)
                changed = True

    # Now C_candidate should be a valid Nikodym complement
    # But it might not be maximal.

    # Let's also try starting from the parabola and adding points.
    C_parabola = set()
    for x in F_q:
        y = mul(x, x)
        C_parabola.add(point_to_idx[(x, y)])

    # Try to add points to C_parabola
    C_improved = set(C_parabola)
    for x_idx in range(q * q):
        if x_idx in C_improved:
            continue
        # Check if adding x_idx keeps the set valid
        C_test = C_improved | {x_idx}
        # Check all points in C_test
        valid = True
        for y_idx in C_test:
            has_free_line = False
            for line in lines_through_point[y_idx]:
                if line & C_test == {y_idx}:
                    has_free_line = True
                    break
            if not has_free_line:
                valid = False
                break
        if valid:
            C_improved = C_test

    # Use the larger of C_candidate and C_improved
    if len(C_improved) > len(C_candidate):
        C_final = C_improved
    else:
        C_final = C_candidate

    # Construct N = complement of C_final
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in C_final:
            (xa, xb), (ya, yb) = pt
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
