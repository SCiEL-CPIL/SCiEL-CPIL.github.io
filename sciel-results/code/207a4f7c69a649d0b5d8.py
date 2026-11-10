# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array.
    """
    start_time = time.time()
    deadline = 600.0
    
    p = 7
    k = 3 # alpha^2 = 3

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if time.time() > deadline - 10:
                        break
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    if time.time() > deadline - 10:
        break

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
