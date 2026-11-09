import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    The complement S is the parabola y = x^2 in F_49^2.
    S is a cap set (no 3 collinear points), so for any x in S, there is a line
    through x intersecting S only at x. For any x not in S, there is a line
    through x disjoint from S.
    Thus N = F_49^2 \ S is a valid Nikodym set.
    |S| = 49, so score is 49.
    """
    p = 7
    k = 3 # alpha^2 = 3

    # Generate all points in F_49^2
    # Point is [[xa, xb], [ya, yb]]
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
    
    # Generate S: parabola y = x^2
    # x = xa + xb * alpha
    # x^2 = (xa^2 + xb^2 * k) + (2 * xa * xb) * alpha
    # y_a = xa^2 + xb^2 * k
    # y_b = 2 * xa * xb
    S = set()
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + xb * xb * k) % p
            yb = (2 * xa * xb) % p
            S.add((xa, xb, ya, yb))
    
    # N is all points not in S
    N = []
    for pt in all_points:
        if pt not in S:
            N.append([list(pt[:2]), list(pt[2:])])
            
    return np.array(N, dtype=np.int64)
