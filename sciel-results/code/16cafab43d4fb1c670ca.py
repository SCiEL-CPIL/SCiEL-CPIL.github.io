import numpy as np

DIMENSIONS = [2, 3, 5]


def construct():
    """
    Returns a rank-28 tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix.
    The decomposition uses two disjoint Strassen 2x2x2 blocks (for j in {0,1} and k in {0,1}
    and for j in {0,1} and k in {2,3}) and schoolbook terms for the remaining (j, k) pairs.
    
    The output format follows the official trace convention:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0

    def add_strassen_block(j_off, k_off):
        nonlocal c
        # Strassen's 7 multiplications for the sub-matrix:
        # A_sub = A[j_off:j_off+2, :]  (rows j_off, j_off+1)
        # B_sub = B[:, k_off:k_off+2]  (cols k_off, k_off+1)
        
        # m1 = (a00 + a11) * (b00 + b11)
        U[0*m + j_off, c] = 1
        U[1*m + (j_off + 1), c] = 1
        V[j_off*p + k_off, c] = 1
        V[(j_off + 1)*p + (k_off + 1), c] = 1
        W[k_off*n + 0, c] = 1
        W[(k_off + 1)*n + 1, c] = 1
        c += 1

        # m2 = (a10 + a11) * b00
        U[1*m + j_off, c] = 1
        U[1*m + (j_off + 1), c] = 1
        V[j_off*p + k_off, c] = 1
        W[k_off*n + 1, c] = 1
        W[(k_off + 1)*n + 1, c] = -1
        c += 1

        # m3 = a00 * (b01 - b11)
        U[0*m + j_off, c] = 1
        V[j_off*p + (k_off + 1), c] = 1
        V[(j_off + 1)*p + (k_off + 1), c] = -1
        W[(k_off + 1)*n + 0, c] = 1
        W[(k_off + 1)*n + 1, c] = 1
        c += 1

        # m4 = a11 * (b10 - b00)
        U[1*m + (j_off + 1), c] = 1
        V[(j_off + 1)*p + k_off, c] = 1
        V[j_off*p + k_off, c] = -1
        W[k_off*n + 0, c] = 1
        W[k_off*n + 1, c] = 1
        c += 1

        # m5 = (a00 + a01) * b11
        U[0*m + j_off, c] = 1
        U[0*m + (j_off + 1), c] = 1
        V[(j_off + 1)*p + (k_off + 1), c] = 1
        W[k_off*n + 0, c] = -1
        W[(k_off + 1)*n + 0, c] = 1
        c += 1

        # m6 = (a10 - a00) * (b00 + b01)
        U[1*m + j_off, c] = 1
        U[0*m + j_off, c] = -1
        V[j_off*p + k_off, c] = 1
        V[j_off*p + (k_off + 1), c] = 1
        W[(k_off + 1)*n + 1, c] = 1
        c += 1

        # m7 = (a01 - a11) * (b10 + b11)
        U[0*m + (j_off + 1), c] = 1
        U[1*m + (j_off + 1), c] = -1
        V[(j_off + 1)*p + k_off, c] = 1
        V[(j_off + 1)*p + (k_off + 1), c] = 1
        W[k_off*n + 0, c] = 1
        c += 1

    # Strassen Block 1: j in {0, 1}, k in {0, 1}
    add_strassen_block(0, 0)

    # Strassen Block 2: j in {0, 1}, k in {2, 3}
    add_strassen_block(0, 2)

    # Schoolbook for remaining (j, k) pairs
    # We need to cover all i, j, k where not (j in {0,1} and k in {0,1,2,3})
    # The only j not covered by Strassen is j=2.
    # So for j=2, all k in {0,1,2,3,4} need schoolbook terms for all i in {0,1}.
    # That is 2 * 5 = 10 terms.
    # Total rank = 7 (block1) + 7 (block2) + 10 (schoolbook) = 24?
    # Wait, the previous analysis said 28. Let's re-verify.
    # Previous code:
    # "Remaining 14 schoolbook terms"
    # Loop: for i in range(n): for j in range(m): for k in range(p):
    #   if j in (0, 1) and k in (0, 1, 2, 3): continue
    #   ...
    # j=0, k=4 -> 2 terms (i=0,1)
    # j=1, k=4 -> 2 terms (i=0,1)
    # j=2, k=0 -> 2 terms
    # j=2, k=1 -> 2 terms
    # j=2, k=2 -> 2 terms
    # j=2, k=3 -> 2 terms
    # j=2, k=4 -> 2 terms
    # Total = 2+2+2+2+2+2+2 = 14 terms.
    # Total rank = 7 + 7 + 14 = 28. Correct.

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1

    # c should be 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
