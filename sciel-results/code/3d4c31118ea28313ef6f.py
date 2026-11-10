# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # We construct the complement C with |C| = q + 1 = 50 points.
    # For any set C with |C| <= q + 1, each point P in C has at most |C|-1 <= q
    # other points, which block at most q of the q+1 available directions.
    # Thus at least one direction remains where P is the unique point of C on that line.
    # So N = F_q^2 \ C is a valid Nikodym set.
    
    # Pick 50 distinct points as C. We'll use the first 50 in lex order.
    c_points = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    c_points.add((xa, xb, ya, yb))
                    if len(c_points) == q + 1:
                        break
                if len(c_points) == q + 1:
                    break
            if len(c_points) == q + 1:
                break
        if len(c_points) == q + 1:
            break
    
    # N is all points NOT in C
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in c_points:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
