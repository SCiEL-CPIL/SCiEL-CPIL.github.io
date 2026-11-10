# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by taking the complement of the parabola C.
    C = {(x, x^2) : x in F_49}.
    Size of C is 49. Size of N is 2401 - 49 = 2352.
    Complement size is 49.
    """
    p = 7
    k = 3  # alpha^2 = 3
    
    # Precompute squares in F_49
    # x = xa + xb * alpha
    # x^2 = (xa + xb*alpha)^2 = xa^2 + 2*xa*xb*alpha + xb^2*alpha^2
    # = (xa^2 + 3*xb^2) + (2*xa*xb)*alpha
    # All mod 7
    
    points_in_C = set()
    for xa in range(p):
        for xb in range(p):
            # Calculate x^2
            y_a = (xa * xa + 3 * xb * xb) % p
            y_b = (2 * xa * xb) % p
            points_in_C.add((xa, xb, y_a, y_b))
            
    points_in_N = []
    # Iterate over all points in F_49^2
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in points_in_C:
                        points_in_N.append([[xa, xb], [ya, yb]])
                        
    return np.array(points_in_N, dtype=np.int64)

# EVOLVE-BLOCK-END
