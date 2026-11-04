"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    start_time = time.time()
    
    # Try rank 17 first
    result = _try_rank17(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank17(n, m, p):
    """Try to construct a rank-17 decomposition for 2x3 x 3x3 matrix multiplication."""
    if n != 2 or m != 3 or p != 3:
        return None
    
    rank = 17
    # U: 6 x 17, V: 9 x 17, W: 6 x 17
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Construction: 9 terms for row 0 (i=0), 8 terms for row 1 (i=1)
    # The 9th term for row 1 is absorbed by modifying one of the shared terms
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Terms 0-8: Standard schoolbook for row 0 (i=0)
    # For each j, k: U[0*3+j, t] = 1, V[j*3+k, t] = 1, W[k*2+0, t] = 1
    t = 0
    for j in range(3):
        for k in range(3):
            U[j][t] = 1.0
            V[j * 3 + k][t] = 1.0
            W[k * 2 + 0][t] = 1.0
            t += 1
    
    # Terms 9-16: For row 1 (i=1), we need 9 outputs but only have 8 terms
    # We need to find a way to compute all 9 products of A1*B with 8 rank-1 terms
    # combined with the existing terms
    
    # Strategy: Use 8 terms where one term contributes to two different (j,k) pairs
    # for row 1, while not disturbing row 0's correctness
    
    # For row 1, the required condition is:
    # sum_r W[k*2+1, r] * U[3+j, r] * V[j*3+k, r] = delta(j, j') for all j, j', k
    
    # With 8 terms (indices 9-16), we have 8 unknowns per (j,k) pair for row 1
    # But we also have U[3+j, r] and V[j*3+k, r] which are shared with the constraint
    
    # Let me try a specific construction:
    # Term 9-16 handle row 1 with a modified pattern
    
    # Actually, let me try: use the fact that for the 3rd column of row 1,
    # we can express one product in terms of others
    
    # Term 9: (j=0, k=0) for row 1
    U[3 + 0][9] = 1.0
    V[0 * 3 + 0][9] = 1.0
    W[0 * 2 + 1][9] = 1.0
    
    # Term 10: (j=0, k=1) for row 1
    U[3 + 0][10] = 1.0
    V[0 * 3 + 1][10] = 1.0
    W[1 * 2 + 1][10] = 1.0
    
    # Term 11: (j=0, k=2) for row 1
    U[3 + 0][11] = 1.0
    V[0 * 3 + 2][11] = 1.0
    W[2 * 2 + 1][11] = 1.0
    
    # Term 12: (j=1, k=0) for row 1
    U[3 + 1][12] = 1.0
    V[1 * 3 + 0][12] = 1.0
    W[0 * 2 + 1][12] = 1.0
    
    # Term 13: (j=1, k=1) for row 1
    U[3 + 1][13] = 1.0
    V[1 * 3 + 1][13] = 1.0
    W[1 * 2 + 1][13] = 1.0
    
    # Term 14: (j=1, k=2) for row 1
    U[3 + 1][14] = 1.0
    V[1 * 3 + 2][14] = 1.0
    W[2 * 2 + 1][14] = 1.0
    
    # Term 15: (j=2, k=0) for row 1
    U[3 + 2][15] = 1.0
    V[2 * 3 + 0][15] = 1.0
    W[0 * 2 + 1][15] = 1.0
    
    # Term 16: (j=2, k=1) for row 1 - but we're missing (j=2, k=2)
    U[3 + 2][16] = 1.0
    V[2 * 3 + 1][16] = 1.0
    W[1 * 2 + 1][16] = 1.0
    
    # We're missing the (j=2, k=2) term for row 1. 
    # We need to absorb it into one of the existing terms.
    
    # This simple construction doesn't work. Let me verify and if it fails, return None.
    result = _verify(U, V, W, n, m, p)
    if result is not None:
        return result
    
    # Try a different construction: modify some terms to absorb the missing product
    # Use a Strassen-like trick on a 2x2 subblock
    
    # Let me try: for the (j=2, k=1) and (j=2, k=2) terms, use a combined approach
    # Reset and try again with a different pattern
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Terms 0-8: Standard for row 0
    t = 0
    for j in range(3):
        for k in range(3):
            U[j][t] = 1.0
            V[j * 3 + k][t] = 1.0
            W[k * 2 + 0][t] = 1.0
            t += 1
    
    # For row 1, use 8 terms with a shared product trick
    # Use (a+b)(c+d) = ac + ad + bc + bd style on a pair
    
    # Term 9: (j=0, k=0)
    U[3][9] = 1.0
    V[0][9] = 1.0
    W[1][9] = 1.0  # W[k=0, i=1]
    
    # Term 10: (j=0, k=1)  
    U[3][10] = 1.0
    V[1][10] = 1.0  # V[j=0, k=1]
    W[3][10] = 1.0  # W[k=1, i=1]
    
    # Term 11: (j=0, k=2)
    U[3][11] = 1.0
    V[2][11] = 1.0  # V[j=0, k=2]
    W[5][11] = 1.0  # W[k=2, i=1]
    
    # Term 12: (j=1, k=0)
    U[4][12] = 1.0
    V[3][12] = 1.0  # V[j=1, k=0]
    W[1][12] = 1.0
    
    # Term 13: (j=1, k=1)
    U[4][13] = 1.0
    V[4][13] = 1.0  # V[j=1, k=1]
    W[3][13] = 1.0
    
    # Term 14: (j=1, k=2)
    U[4][14] = 1.0
    V[5][14] = 1.0  # V[j=1, k=2]
    W[5][14] = 1.0
    
    # Term 15: (j=2, k=0)
    U[5][15] = 1.0
    V[6][15] = 1.0  # V[j=2, k=0]
    W[1][15] = 1.0
    
    # Terms 16: Need to cover (j=2, k=1) and (j=2, k=2) with one term
    # This is impossible with a single rank-1 term unless we use the structure
    
    # This approach isn't working. Let me return None to fall back to rank 18.
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct. Returns the dict if valid, None otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return None
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return None
    
    # Check: for all i, j1, j2, k: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == delta(j1,j2)
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(s - expected) > 1e-9:
                        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
