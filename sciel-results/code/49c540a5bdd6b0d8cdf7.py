import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # Field element representation: (a, b) means a + b*alpha
    # Precompute field operations
    def fadd(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def fmul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)

    def fsub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def fneg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def finv(a):
        a0, a1 = a
        norm = (a0 * a0 - a1 * a1 * k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p - 2, p)
        return ((a0 * norm_inv) % p, (-a1 * norm_inv) % p)

    # All field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}

    # Precompute multiplication table
    mul_table = {}
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            mul_table[(i, j)] = field_elements.index(fmul(a, b))

    # Precompute all lines
    # A line is identified by (point, direction) where direction is (1, m) or (0, 1)
    # But we need to be careful: lines with direction (1, m) for different m are different
    # Also, the same geometric line can be represented by different (point, direction) pairs
    # We want to identify unique lines

    # For a line with direction (1, m): points are {(x0 + t, y0 + t*m) : t in F_49}
    # For a line with direction (0, 1): points are {(x0, y0 + t) : t in F_49} (vertical lines)

    # Let's enumerate all lines:
    # Vertical lines: 49 lines, each with x fixed
    # Non-vertical lines: for each m in F_49 and each "intercept" (y when x=0), that's 49*49 = 2401
    # But wait, for direction (1, m), the line is determined by m and the y-intercept (value of y when x=0)
    # Actually, for direction (1, m), a line can be written as y = m*x + c for some c in F_49
    # So there are 49 choices for m and 49 choices for c, giving 2401 non-vertical lines
    # Plus 49 vertical lines, total 2450 lines

    # Let's build a list of all lines, each as a set of 49 points
    # But for efficiency, we'll represent lines more compactly

    # Actually, let's think about this differently.
    # We want to find a set N such that for every point x, there's a line through x
    # (with direction (1, m) or (0, 1)) such that all other points on that line are in N.

    # This is equivalent to: for every point x, there exists a line L through x
    # (with allowed direction) such that L \ {x} subset of N.
    # Equivalently, L \ N = {x} or L \ N = empty set (if x in N, then L\N could be empty)
    # Wait, no. If x in N, then we need L \ {x} subset of N, which means L subset of N.
    # If x not in N, then we need L \ {x} subset of N, which means L \ N = {x}.

    # So for each point x:
    # - If x in N: there must be an allowed line L through x with L subset of N
    # - If x not in N: there must be an allowed line L through x with L \ N = {x}

    # Let's try a simple construction.
    # Consider N = F_49^2 \ S where S is the set of points on a parabola y = x^2.
    # |S| = 49, so |N| = 2401 - 49 = 2352.

    # Let's check if this works.
    # For a point x = (a, b) in F_49^2:
    # - If x in S (i.e., b = a^2): we need a line through x with all points in N.
    #   But x is in S, not N. So we need L \ {x} subset of N, i.e., L \ S = {x}.
    #   The tangent line to y = x^2 at x = a has direction (1, 2a).
    #   The tangent line is y = 2a*x - a^2.
    #   Does this line intersect the parabola only at x = a?
    #   In characteristic 7, the equation x^2 = 2a*x - a^2 becomes x^2 - 2a*x + a^2 = 0,
    #   which is (x - a)^2 = 0, so it has a double root at x = a.
    #   So the tangent line intersects the parabola only at x = a (with multiplicity 2).
    #   Thus L \ S = {x}, which is what we need.

    # - If x not in S: we need a line through x with L \ N = {x}, i.e., L \ S = {x}.
    #   Wait, that's the same condition. But x is not in S, so L \ S = {x} means x in S, contradiction.
    #   Let me re-read the condition.
    #   "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N"
    #   So for x not in N (i.e., x in S), we need L \ {x} subset of N, i.e., L \ S = {x}.
    #   For x in N (i.e., x not in S), we need L \ {x} subset of N, i.e., L \ S = empty set (since x not in S).
    #   Wait, L \ {x} subset of N means all points in L except x are in N.
    #   If x in N, then L \ {x} subset of N means L subset of N, i.e., L \ S = empty set.
    #   If x not in N (x in S), then L \ {x} subset of N means L \ S = {x}.

    # So for x in N (x not on parabola): need a line through x with no points on the parabola.
    # For x in S (x on parabola): need a line through x with exactly one point on the parabola (namely x).

    # For x in S: the tangent line works as argued above.
    # For x in N: we need a line through x that doesn't intersect the parabola at all.
    # A line y = m*x + c intersects the parabola y = x^2 when x^2 = m*x + c, i.e., x^2 - m*x - c = 0.
    # This quadratic has 0, 1, or 2 roots in F_49.
    # We need a line through x = (a, b) (where b != a^2) such that the quadratic has no roots.
    # The line through (a, b) with slope m is y - b = m*(x - a), i.e., y = m*x + (b - m*a).
    # So c = b - m*a.
    # The quadratic is x^2 - m*x - (b - m*a) = 0, i.e., x^2 - m*x + m*a - b = 0.
    # The discriminant is m^2 - 4*(m*a - b) = m^2 - 4*m*a + 4*b.
    # We need this discriminant to be a non-square in F_49 (including 0? No, 0 gives a double root).
    # Actually, we need the quadratic to have no roots, so the discriminant must be a non-square (and nonzero).

    # For a given x = (a, b) with b != a^2, can we find m such that m^2 - 4*m*a + 4*b is a non-square?
    # As m varies over F_49, the expression m^2 - 4*a*m + 4*b takes 49 values.
    # It's a quadratic in m, so it takes at most 25 distinct values (since the map m -> m^2 is 2-to-1 except at 0).
    # Actually, the range of a non-constant quadratic over a field of odd characteristic has size (q+1)/2 = 25.
    # Of these 25 values, some are squares, some are non-squares, and possibly one is 0.
    # Since 49 is odd, and the quadratic is not constant, it must take both square and non-square values.
    # So there exists m such that the discriminant is a non-square, giving a line with no intersection with the parabola.

    # Therefore, N = F_49^2 \ {parabola} is a valid Nikodym set with |N| = 2401 - 49 = 2352.

    # Let's construct it.
    S = set()
    for x in field_elements:
        x2 = fmul(x, x)
        S.add((x, x2))

    # Build N as the complement
    points = []
    for x in field_elements:
        for y in field_elements:
            if (x, y) not in S:
                xa, xb = x
                ya, yb = y
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
