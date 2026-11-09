# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    We use the set N = F_q^2 \ U, where U is the curve defined by
    Trace(x) + Norm(y) = 0 in F_49^2.
    
    U is defined by 2*x_a + y_a^2 - 3*y_b^2 = 0 (mod 7).
    Since this is a quadratic curve in the affine plane (over the field extension structure),
    any line intersects U in at most 2 points.
    For any point x in U, there exists a tangent line that intersects U only at x.
    Thus, for every x in F_q^2, there is a line through x contained in N U {x}.
    The size of the complement S = U is 343.
    """
    p = 7
    k = 3  # alpha^2 = 3

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if (x, y) is in U
                    # 2*xa + ya^2 - k*yb^2 == 0 (mod p)
                    if (2 * xa + ya * ya - k * yb * yb) % p != 0:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
