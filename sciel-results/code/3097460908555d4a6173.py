# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2. The complement C is the set of points in the
    Hermitian unital U = {(x,y) : Trace(x) + Norm(y) = 0} whose y-coordinate is not
    in the special set Y = {(u^2 mod 7, v) : u,v in F_7}. This gives |C| = 147,
    so |N| = 2401 - 147 = 2254.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7

    # Special y-coordinate set Y = {(u^2 mod 7, v)}: y_a must be a square mod 7.
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if time.time() - start_time > deadline:
                        break
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue  # excluded point
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
