"""4x4 matrix multiplication tensor decomposition using recursive Strassen."""

import time

DIMENSIONS = [4, 4, 4]
DEADLINE = time.time() + 145  # 5s safety margin

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using an explicit enumeration of the
    49 terms from the recursive Strassen algorithm.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Base Strassen 2x2 factors
    # Format: list of (coeff, row, col)
    sU = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
    ]
    sV = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (1, 1, 1)],
    ]
    sW = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (-1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 0, 0), (1, 1, 0)],
        [(-1, 0, 0), (1, 0, 1)],
        [(1, 1, 1)],
        [(1, 0, 0)],
    ]
    
    # Outer Strassen factors (identical structure to inner)
    oU = sU
    oV = sV
    oW = sW
    
    col = 0
    for o in range(7):
        if time.time() > DEADLINE:
            # If we run out of time, return whatever we have (which should be complete for this small size)
            # But to be safe, we break and return.
            break
        for i in range(7):
            # Fill U
            for (oc, obr, obc) in oU[o]:
                for (ic, ir, icl) in sU[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    U[gr * m + gc][col] += oc * ic
            
            # Fill V
            for (oc, obr, obc) in oV[o]:
                for (ic, ir, icl) in sV[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    V[gr * p + gc][col] += oc * ic
            
            # Fill W
            for (oc, obr, obc) in oW[o]:
                for (ic, ir, icl) in sW[i]:
                    gr = 2 * obr + ir
                    gc = 2 * obc + icl
                    W[gc * n + gr][col] += oc * ic
            
            col += 1
            if time.time() > DEADLINE:
                break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
