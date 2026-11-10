# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49.
    We choose N to be the complement of a single line L.
    N = F_q^2 \ L.
    |N| = q^2 - q.
    Complement S = L, |S| = q = 49.
    
    Validity check:
    1. For x in N, there exists a line in N through x.
       Lines parallel to L are in N. Since x is not in L, x is on exactly one line parallel to L.
       This line is fully in N. So condition holds.
    2. For x in S (i.e., x in L), there exists a line through x with intersection S = {x}.
       Any line through x other than L intersects L only at x.
       So L' \cap S = {x}. Condition holds.
       
    Score = 49.
    """
    p = 7
    q = p * p
    
    # Define the line L: y = 0 (i.e., y_a = 0, y_b = 0)
    # Points in L are [[xa, xb], [0, 0]] for xa, xb in 0..6
    # S = L
    # N = F_q^2 \ L
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # If point is in L, skip it (it's in S)
                    if ya == 0 and yb == 0:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
