# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    The complement size is maximized by the specific algebraic construction.
    """
    p = 7
    k = 3  # Smallest quadratic non-residue mod 7
    
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
                    # Check if in Unital U: Tr(x) + Norm(y) = 0
                    # Tr((xa, xb)) = 2*xa
                    # Norm((ya, yb)) = ya^2 - yb^2*k
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    
                    # Exclude points in U but not in B (where B is U intersect Y x F_q)
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])
                    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
