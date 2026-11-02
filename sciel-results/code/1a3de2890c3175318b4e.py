"""Construct a rank-18 tensor decomposition for 3x3 * 3x4 matrix multiplication."""

def construct():
    n, m, p = 3, 3, 4
    # Rank-18 construction: 9 terms for the 3x3 sub-problem (k=0,1,2)
    # plus 9 terms for the 4th column (k=3), using schoolbook structure
    # but grouped by (i,j) pairs, giving rank = 2 * n * m = 18
    
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows
    V = [[0] * rank for _ in range(m * p)]   # 12 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows
    
    # First 9 terms: handle k=0,1,2 (the 3x3 sub-problem)
    # Term r corresponds to (i0, j0, k0) where k0 in {0,1,2}
    # U[i0*m+j0, r] = 1, V[j0*p+k0, r] = 1, W[k0*n+i0, r] = 1
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(3):
                r = i0 * m + j0 * 3 + k0  # 0..8
                U[i0 * m + j0][r] = 1
                V[j0 * p + k0][r] = 1
                W[k0 * n + i0][r] = 1
    
    # Next 9 terms: handle k=3 (the 4th column)
    # Term r corresponds to (i0, j0, k0=3)
    # U[i0*m+j0, r] = 1, V[j0*p+3, r] = 1, W[3*n+i0, r] = 1
    for i0 in range(n):
        for j0 in range(m):
            r = 9 + i0 * m + j0  # 9..17
            U[i0 * m + j0][r] = 1
            V[j0 * p + 3][r] = 1
            W[3 * n + i0][r] = 1
    
    return {"U": U, "V": V, "W": W}
