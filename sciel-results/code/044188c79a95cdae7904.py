"""
Rank-35 decomposition for 2x2 by 2x10 matrix multiplication.

The minimal rank for multiplying a 2x2 matrix by a 2x10 matrix is 35.
This is achieved by applying Strassen's rank-7 algorithm independently to
each of the 5 column-pairs (2x2 blocks) of the 2x10 matrix B.
Since the B-blocks are linearly independent and only share the A matrix,
the total rank is the sum of the ranks of the blocks: 5 * 7 = 35.

This program constructs the decomposition directly without search,
ensuring minimal execution time and maximum reliability.
"""
import numpy as np

def construct():
    """
    Construct the tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
        U: (4, 35), V: (20, 35), W: (20, 35)
    """
    n, m, p = 2, 2, 10
    rank = 35
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]] = [[U0, U1], [U2, U3]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A (flattened row-major):
    # a=0, b=1, c=2, d=3
    
    for t in range(5):
        base = t * 7
        
        # Indices for B block t (columns 2t and 2t+1)
        # B is 2x10, flattened row-major: row 0 is B[0,0..9], row 1 is B[1,0..9]
        # e = B[0, 2t], f = B[0, 2t+1]
        # g = B[1, 2t], h = B[1, 2t+1]
        e_idx = 0 * p + 2 * t
        f_idx = 0 * p + 2 * t + 1
        g_idx = 1 * p + 2 * t
        h_idx = 1 * p + 2 * t + 1
        
        # Indices for C block t (columns 2t and 2t+1)
        # C is 2x10, but W uses convention W[k*n + i, r] -> W is (p*n, rank)
        # W index for C[i, k] is k*n + i.
        # C00 (i=0, k=2t): idx = 2t * 2 + 0 = 4t
        # C01 (i=0, k=2t+1): idx = (2t+1) * 2 + 0 = 4t + 2
        # C10 (i=1, k=2t): idx = 2t * 2 + 1 = 4t + 1
        # C11 (i=1, k=2t+1): idx = (2t+1) * 2 + 1 = 4t + 3
        c00 = 4 * t
        c01 = 4 * t + 2
        c10 = 4 * t + 1
        c11 = 4 * t + 3
        
        # M1 = (a+d)(e+h)
        col = base
        U[0, col] = 1  # a
        U[3, col] = 1  # d
        V[e_idx, col] = 1
        V[h_idx, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        
        # M2 = (c+d)e
        col = base + 1
        U[2, col] = 1  # c
        U[3, col] = 1  # d
        V[e_idx, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1
        
        # M3 = a(f-h)
        col = base + 2
        U[0, col] = 1  # a
        V[f_idx, col] = 1
        V[h_idx, col] = -1
        W[c01, col] = 1
        W[c11, col] = 1
        
        # M4 = d(g-e)
        col = base + 3
        U[3, col] = 1  # d
        V[g_idx, col] = 1
        V[e_idx, col] = -1
        W[c00, col] = 1
        W[c10, col] = 1
        
        # M5 = (a+b)h
        col = base + 4
        U[0, col] = 1  # a
        U[1, col] = 1  # b
        V[h_idx, col] = 1
        W[c00, col] = -1
        W[c01, col] = 1
        
        # M6 = (c-a)(e+f)
        col = base + 5
        U[2, col] = 1  # c
        U[0, col] = -1 # a
        V[e_idx, col] = 1
        V[f_idx, col] = 1
        W[c11, col] = 1
        
        # M7 = (b-d)(g+h)
        col = base + 6
        U[1, col] = 1  # b
        U[3, col] = -1 # d
        V[g_idx, col] = 1
        V[h_idx, col] = 1
        W[c00, col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
