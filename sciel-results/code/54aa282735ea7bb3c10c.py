import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the complement of the Hermitian unital.
    The Hermitian unital U = {(x, y) in F_49^2 : x^7 + x = y^8} is a set of 344 points
    such that every point in U lies on a line that intersects U in exactly one point (the tangent).
    Thus, N = F_49^2 \ U is a Nikodym set.
    """
    p = 7
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    
    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + b1*a2)*alpha + b1*b2*alpha^2
        # alpha^2 = 3
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + b1 * a2) % p
        return (a, b)
    
    def pow_field(e, n):
        result = (1, 0)
        base = e
        while n > 0:
            if n % 2 == 1:
                result = mul(result, base)
            base = mul(base, base)
            n //= 2
        return result
    
    # Generate all elements of F_49
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Find points in the unital U
    unital_points = set()
    for x in elements:
        x7 = pow_field(x, 7)
        lhs = add(x7, x)
        for y in elements:
            y8 = pow_field(y, 8)
            if lhs == y8:
                unital_points.add((x, y))
    
    # Construct N = F_49^2 \ U
    points = []
    for x in elements:
        for y in elements:
            if (x, y) not in unital_points:
                # Convert (x, y) to [[xa, xb], [ya, yb]]
                points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)
