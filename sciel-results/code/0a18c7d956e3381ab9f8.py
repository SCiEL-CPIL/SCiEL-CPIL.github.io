# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    The set N is the complement of the line y = 0.
    C = {(x, 0) : x in F_49}.
    For any x in C, the vertical line through x intersects C only at x.
    For any y in N, the horizontal line through y is contained in N.
    Thus, N is a valid Nikodym set.
    |N| = 49^2 - 49 = 2352.
    """
    p = 7
    q = p * p
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Exclude points on the line y = 0 (ya = 0, yb = 0)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
