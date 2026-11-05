import numpy as np
import time

DIMENSIONS = [3, 3, 7]
DEADLINE = 120.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition for matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Compute the tensor T
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # T: (nm, mp, pn)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Create target tensor
    Target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Target[a, b, c] = 1.0
                
    return np.allclose(T, Target, atol=tol)

def _schoolbook_block(n, m, p, offset_U, offset_V, offset_W, U, V, W, col_start):
    """Fill schoolbook decomposition for a block of size n x m x p starting at col_start."""
    col = col_start
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j + offset_U, col] = 1
                V[j * p + k + offset_V, col] = 1
                W[k * n + i + offset_W, col] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Strategy: Split B (m x p) into blocks.
    # p = 7. Split into 3, 3, 1.
    # Block 1: 3x3x3
    # Block 2: 3x3x3
    # Block 3: 3x3x1
    
    rank1 = n * m * 3  # 9
    rank2 = n * m * 3  # 9
    rank3 = n * m * 1  # 9
    total_rank = rank1 + rank2 + rank3 # 27
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: k in 0..2
    # U offset: 0
    # V offset: 0 (j*p + k, k is 0..2)
    # W offset: 0 (k*n + i, k is 0..2)
    col = _schoolbook_block(n, m, 3, 0, 0, 0, U, V, W, 0)
    
    # Block 2: k in 3..5
    # U offset: 0
    # V offset: 0 (j*p + k, k is 3..5 -> j*7 + 3..5)
    # W offset: 3*n (k*n + i, k is 3..5 -> 3*n + i)
    col = _schoolbook_block(n, m, 3, 0, 0, 3*n, U, V, W, col)
    
    # Block 3: k in 6..6
    # U offset: 0
    # V offset: 0 (j*p + 6)
    # W offset: 6*n (6*n + i)
    col = _schoolbook_block(n, m, 1, 0, 0, 6*n, U, V, W, col)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to full schoolbook if something went wrong
        rank_full = n * m * p
        U = np.zeros((n * m, rank_full))
        V = np.zeros((m * p, rank_full))
        W = np.zeros((p * n, rank_full))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    U = U.tolist()
                    V = V.tolist()
                    W = W.tolist()
                    return {"U": U, "V": V, "W": W}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
