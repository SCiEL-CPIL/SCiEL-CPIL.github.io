"""Matrix multiplication 4x4x9: Kronecker product of two 2x2x3 rank-7 decompositions."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _winograd_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication with rank 7.
    
    U: (4, 7) for A (2x2), indexed U[i*2+j, r]
    V: (6, 7) for B (2x3), indexed V[j*3+k, r]
    W: (6, 7) for C (2x3), indexed W[k*2+i, r]
    """
    U = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0]
    ], dtype=np.float64)
    
    V = np.array([
        [1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1, 1],
        [1,  1, -1, -1, 1,  1, -1],
        [1, -1, -1,  1, 1, -1, -1],
        [1,  1,  1,  1, -1, -1, -1],
        [1, -1,  1, -1, -1, 1, -1]
    ], dtype=np.float64)
    
    W = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  1,  1],
        [0,  0,  0,  0,  1, -1, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _rank12_2x2x9():
    """Rank-12 decomposition for 2x2x9 matrix multiplication.
    
    Splits the 9 columns into 3 groups of 3, uses rank-7 for each 2x2x3 block.
    Total rank = 3 * 7 = 21? No, that's not right.
    
    Actually, for 2x2x9, we can use a more clever approach.
    The 2x2x9 multiplication tensor can be decomposed using Winograd-style methods.
    
    For 2x2x9: A is 2x2, B is 2x9, C is 2x9.
    We can think of B as having 9 columns. We can group them into 3 groups of 3.
    Each group gives a 2x2x3 problem with rank 7.
    But the U matrix for A is shared across all groups.
    
    Actually, let me reconsider. The standard approach:
    - A is 2x2 (4 elements)
    - B is 2x9 (18 elements)  
    - C is 2x9 (18 elements)
    
    For each column group of 3 in B (and corresponding columns in C), we have a 2x2x3 problem.
    The U factors for A are the same for all three groups.
    
    So: U is (4, 7) - same for all three column groups
    V is (18, 21) - 3 blocks of 6x7
    W is (18, 21) - 3 blocks of 6x7
    
    Wait, that gives rank 21, not 12.
    
    Let me think differently. For 2x2x9, we can use a different decomposition.
    
    Actually, the rank of the 2x2xn multiplication tensor is known to be about 2n+2 for large n.
    For n=9, that would be about 20.
    
    Let me just use the block approach: 3 separate 2x2x3 decompositions, each rank 7, total rank 21.
    """
    U1, V1, W1 = _winograd_2x2x3()
    
    rank = 21
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((18, rank), dtype=np.float64)
    W = np.zeros((18, rank), dtype=np.float64)
    
    for block in range(3):
        r_offset = block * 7
        # U is the same for all blocks
        U[:, r_offset:r_offset+7] = U1
        # V: rows j*9+k where j=0,1, k=0..8
        # For block b, columns are 3*b, 3*b+1, 3*b+2
        for j in range(2):
            for k in range(3):
                v_idx = j * 9 + (3 * block + k)
                V[v_idx, r_offset:r_offset+7] = V1[j*3+k, :]
        # W: rows k*2+i where k=0..8, i=0,1
        # For block b, rows are 3*b, 3*b+1, 3*b+2 (as row indices in C)
        for k in range(3):
            for i in range(2):
                w_idx = (3 * block + k) * 2 + i
                W[w_idx, r_offset:r_offset+7] = W1[k*2+i, :]
    
    return U, V, W

def _rank9_1x1x9():
    """Rank-9 decomposition for 1x1x9 matrix multiplication.
    
    A is 1x1 (scalar a), B is 1x9 (vector b), C is 1x9 (vector c = a*b).
    This is just scalar-vector multiplication.
    Rank = 9 (one per output element).
    
    U: (1, 9) - U[0, r] = 1 for all r
    V: (9, 9) - V[j*9+k, r] where j=0, k=0..8; V[k, r] = delta_{k,r}
    W: (9, 9) - W[k*1+i, r] where k=0..8, i=0; W[k, r] = delta_{k,r}
    """
    rank = 9
    U = np.ones((1, rank), dtype=np.float64)
    V = np.zeros((9, rank), dtype=np.float64)
    W = np.zeros((9, rank), dtype=np.float64)
    
    for k in range(9):
        V[k, k] = 1
        W[k, k] = 1
    
    return U, V, W

def _kron_decompose(n, m, p):
    """Kronecker product of two 2x2x3 Winograd rank-7 decompositions.
    
    MM(2,2,3) x MM(2,2,3) = MM(4,4,9) with rank 7*7 = 49.
    
    A (4x4) viewed as 2x2 blocks of 2x2: i = i1*2+i0, j = j1*2+j0
    B (4x9) viewed as: j = j1*2+j0, k = k1*3+k0  
    C (4x9) viewed as: i = i1*2+i0, k = k1*3+k0
    
    U[i*m+j, r] = U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
    V[j*p+k, r] = V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
    W[k*n+i, r] = W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
    where r = r1*7 + r2
    """
    n, m, p = 4, 4, 9
    
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U: depends on (i, j, r) where i = i1*2+i0, j = j1*2+j0
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U[i*m+j, r] = U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
            
            # V: depends on (j, k, r) where j = j1*2+j0, k = k1*3+k0
            for j1 in range(2):
                for k1 in range(3):
                    for j0 in range(2):
                        for k0 in range(3):
                            j = j1 * 2 + j0
                            k = k1 * 3 + k0
                            V[j*p+k, r] = V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
            
            # W: depends on (k, i, r) where k = k1*3+k0, i = i1*2+i0
            for k1 in range(3):
                for i1 in range(2):
                    for k0 in range(3):
                        for i0 in range(2):
                            k = k1 * 3 + k0
                            i = i1 * 2 + i0
                            W[k*n+i, r] = W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
    
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Kronecker product decomposition
    try:
        U, V, W = _kron_decompose(n, m, p)
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
