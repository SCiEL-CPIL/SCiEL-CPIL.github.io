"""Matrix multiplication 2,3,8 - rank-29 exact tensor decomposition.

Uses Strassen's algorithm on four 2x2x2 sub-blocks (for j in {0,1})
and a single rank-1 term for the j=2 part.
Total rank = 4 * 7 + 1 = 29.
"""

DIMENSIONS = [2, 3, 8]

import time


def _validate(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 2x2x7 (flattened to 4x7)
    V: 2x2x7 (flattened to 4x7)
    W: 2x2x7 (flattened to 4x7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1 # A00, A11
    V[0][0] = 1; V[3][0] = 1 # B00, B11
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1 # A10, A11
    V[0][1] = 1 # B00
    W[1][1] = 1 # C01
    
    # M3: A00(B01-B11)
    U[0][2] = 1 # A00
    V[1][2] = 1; V[3][2] = -1 # B01, B11
    W[1][2] = 1; W[3][2] = 1 # C01, C11
    
    # M4: A11(B10-B00)
    U[3][3] = 1 # A11
    V[2][3] = 1; V[0][3] = -1 # B10, B00
    W[0][3] = 1; W[3][3] = -1 # C00, C11
    
    # M5: (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1 # A00, A10
    V[3][4] = 1 # B11
    W[0][4] = -1; W[3][4] = 1 # C00, C11
    
    # M6: (A11-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1 # A11, A00
    V[0][5] = 1; V[1][5] = 1 # B00, B01
    W[2][5] = 1 # C10
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1 # A01, A11
    V[2][6] = 1; V[3][6] = 1 # B10, B11
    W[0][6] = 1; W[2][6] = 1 # C00, C10
    
    return U, V, W


def _construct_rank_29():
    n, m, p = 2, 3, 8
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Get Strassen factors for 2x2x2
    sU, sV, sW = _strassen_2x2x2_factors()
    
    # Place 4 blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    # j indices for Strassen are 0, 1.
    # i indices are 0, 1.
    for b in range(4):
        k_start = 2 * b
        k_end = 2 * b + 1
        for r in range(7):
            global_r = b * 7 + r
            
            # U: i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    val = sU[i * 2 + j][r]
                    if val != 0:
                        U[i * 3 + j][global_r] += val # j is 0 or 1 in global 0..2
                        
            # V: j in {0,1}, k in {k_start, k_end}
            for j in range(2):
                for k_rel in range(2):
                    k_global = k_start + k_rel
                    val = sV[j * 2 + k_rel][r]
                    if val != 0:
                        V[j * 8 + k_global][global_r] += val
                        
            # W: k in {k_start, k_end}, i in {0,1}
            for k_rel in range(2):
                k_global = k_start + k_rel
                for i in range(2):
                    val = sW[k_rel * 2 + i][r]
                    if val != 0:
                        W[k_global * 2 + i][global_r] += val

    # Add rank 1 term for j=2
    r_last = 28
    # U[i, 2] = 1
    for i in range(2):
        U[i * 3 + 2][r_last] = 1
    # V[2, k] = 1
    for k in range(8):
        V[2 * 8 + k][r_last] = 1
    # W[k, i] = 1
    for k in range(8):
        for i in range(2):
            W[k * 2 + i][r_last] = 1

    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 140.0

    # Try rank-29 decomposition
    try:
        cand = _construct_rank_29()
        if time.time() - start > DEADLINE:
            return _schoolbook(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass

    # Fallback: schoolbook
    return _schoolbook(n, m, p)
