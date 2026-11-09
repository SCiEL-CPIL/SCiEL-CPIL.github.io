# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the Hermitian unital blocking set construction.
    Returns the set N as an array of points.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    p = 7
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3
    
    # Special y-coordinate set Y = {(u^2 mod p, v)}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # Build N: all points NOT in (U ∩ complement of y_special)
    # i.e., exclude points that are in U and have y NOT in y_special
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue  # excluded
                    points.append([[xa, xb], [ya, yb]])
    
    result = np.array(points, dtype=np.int64)
    
    if time.time() > deadline - 10:
        return result
    
    return result


# EVOLVE-BLOCK-END
