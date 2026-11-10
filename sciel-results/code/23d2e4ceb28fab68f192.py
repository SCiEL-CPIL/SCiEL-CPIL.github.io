import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    Returns the complement set S as the Nikodym set N? 
    No, the task says: "returning ... listing the points of a set N".
    "Metric: the complement size 2401 - |N|".
    So we return N.
    The seed returns N.
    """
    p = 7
    q = p * p

    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Special y-coordinate set Y = {(u^2 mod p, v)}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue  # excluded point (belongs to S)
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
