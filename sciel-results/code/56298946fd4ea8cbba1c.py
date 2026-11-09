# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by excluding a maximum-size arc.
    
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[x_a, x_b], [y_a, y_b]] representing (x_a + x_b*alpha, y_a + y_b*alpha).
    
    We find a maximum arc (no three collinear) in F_49^2 and return its complement.
    The maximum arc size in F_q^2 is q+1 for q odd, so we expect 50 points.
    This gives |N| = 2401 - 50 = 2351.
    
    Validity: For any point x:
    - If x is not in the arc, any direction works since at most two points on any line
      through x are in the arc (and x is not one of them), so at least 47 points on 
      that line are in N.
    - If x is in the arc, use a direction such that the line through x in that direction
      contains at most x from the arc. Since the arc has no three collinear, for any
      other point y in the arc, the line through x and y contains exactly x and y from
      the arc. So for any direction not parallel to any line through x and another arc
      point, the line contains only x from the arc, giving 48 points in N.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field arithmetic: elements are (a, b) representing a + b*alpha
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_mul(a, b):
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_inv(a):
        # Inverse of a0 + a1*alpha in F_7[alpha]/(alpha^2 - 3)
        # (a0 + a1*alpha)(b0 + b1*alpha) = 1
        # a0*b0 + a1*b1*3 = 1, a0*b1 + a1*b0 = 0
        # From second: b0 = -a0*b1/a1 (if a1 != 0)
        # Substitute: a0*(-a0*b1/a1) + a1*b1*3 = 1
        # b1 * (-a0^2/a1 + 3*a1) = 1
        # b1 * ((-a0^2 + 3*a1^2)/a1) = 1
        # b1 = a1 / (3*a1^2 - a0^2)
        a0, a1 = a
        denom = (3 * a1 * a1 - a0 * a0) % p
        if denom == 0:
            return (0, 0)  # Should not happen for nonzero element
        # Find inverse of denom in F_7
        for d_inv in range(p):
            if (denom * d_inv) % p == 1:
                break
        b1 = (a1 * d_inv) % p
        b0 = (-a0 * b1 * field_inv((a1, 0))[1] if a1 != 0 else 0) % p
        # Simpler: b0 = -a0 * b1 / a1
        if a1 != 0:
            for a1_inv in range(p):
                if (a1 * a1_inv) % p == 1:
                    break
            b0 = (-a0 * b1 * a1_inv) % p
        else:
            # a1 = 0, so a = a0, inverse is 1/a0
            for a0_inv in range(p):
                if (a0 * a0_inv) % p == 1:
                    break
            return (a0_inv, 0)
        return (b0, b1)
    
    # Generate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    # Check if three points are collinear
    def are_collinear(p1, p2, p3):
        # p1, p2, p3 are ((xa, xb), (ya, yb))
        # Collinear if (p2-p1) and (p3-p1) are linearly dependent
        d1 = (field_sub(p2[0], p1[0]), field_sub(p2[1], p1[1]))
        d2 = (field_sub(p3[0], p1[0]), field_sub(p3[1], p1[1]))
        # d1 and d2 are dependent if d1_x * d2_y - d1_y * d2_x = 0
        cross = field_sub(field_mul(d1[0], d2[1]), field_mul(d1[1], d2[0]))
        return cross == (0, 0)
    
    # Greedily build a maximum arc
    arc = []
    for pt in all_points:
        if len(arc) >= 50:
            break
        valid = True
        for i in range(len(arc)):
            for j in range(i + 1, len(arc)):
                if are_collinear(arc[i], arc[j], pt):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            arc.append(pt)
    
    # Build complement
    arc_set = set(arc)
    complement = []
    for pt in all_points:
        if pt not in arc_set:
            complement.append([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]])
    
    return np.array(complement, dtype=np.int64)


# EVOLVE-BLOCK-END
