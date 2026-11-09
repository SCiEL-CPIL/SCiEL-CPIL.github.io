# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    
    The set N is defined as the complement of S, where S is a subset of the 
    Hermitian unital U = {(x, y) : Tr(x) + Nm(y) = 0}. 
    Specifically, we exclude points in U whose y-coordinate has a non-zero square 
    a-coordinate (in F_p). This construction yields a valid Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, use 590s for safety

    p = 7
    q = p * p

    # Find the quadratic non-residue k mod p such that alpha^2 = k
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Precompute y_special: y = (ya, yb) is special if ya is a NON-ZERO square mod p
    # Non-zero squares in F_7: {1, 2, 4}
    squares = { (u * u) % p for u in range(1, p) }
    y_special = set()
    for ya in range(p):
        for yb in range(p):
            if ya in squares:
                y_special.add((ya, yb))

    points = []
    # Iterate over all points in F_q^2
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if point is in the Unital U
                    # Tr(xa, xb) = 2*xa
                    # Nm(ya, yb) = ya^2 - yb^2*k
                    # Condition: 2*xa + ya^2 - yb^2*k == 0 (mod p)
                    val = (2 * xa + ya * ya - yb * yb * k) % p
                    if val == 0:
                        # If in Unital, check if y is special
                        if (ya, yb) in y_special:
                            # y is special, so this point is in B (blocking set part)
                            # The prompt excludes S = U \ B.
                            # So points in B are NOT in S, so they are in N.
                            points.append([[xa, xb], [ya, yb]])
                        else:
                            # y is not special, so this point is in S (excluded)
                            pass
                    else:
                        # Not in Unital, so in N
                        points.append([[xa, xb], [ya, yb]])
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
