import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
DEADLINE = 115  # seconds (safety margin before 120s kill)

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _build_rank_28(n, m, p):
    """Build the rank-28 decomposition using block structure."""
    U0 = [[0] * 28 for _ in range(n * m)]
    V0 = [[0] * 28 for _ in range(m * p)]
    W0 = [[0] * 28 for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U0[i * m + 2][r] = 1
            V0[2 * p + k][r] = 1
            W0[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U0[i * m + j][r] = 1
            V0[j * p + 4][r] = 1
            W0[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    # M1
    U0[0 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 0][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U0[1 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 0][r] = 1
    W0[0 * n + 1][r] = 1
    W0[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U0[0 * m + 0][r] = 1
    V0[0 * p + 1][r] = 1
    V0[1 * p + 1][r] = -1
    W0[1 * n + 0][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U0[1 * m + 1][r] = 1
    V0[1 * p + 0][r] = 1
    V0[0 * p + 0][r] = -1
    W0[0 * n + 0][r] = 1
    W0[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U0[0 * m + 0][r] = 1
    U0[0 * m + 1][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = -1
    W0[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U0[1 * m + 0][r] = 1
    U0[0 * m + 0][r] = -1
    V0[0 * p + 0][r] = 1
    V0[0 * p + 1][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U0[0 * m + 1][r] = 1
    U0[1 * m + 1][r] = -1
    V0[1 * p + 0][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    # M1
    U0[0 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 2][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U0[1 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 2][r] = 1
    W0[2 * n + 1][r] = 1
    W0[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U0[0 * m + 0][r] = 1
    V0[0 * p + 3][r] = 1
    V0[1 * p + 3][r] = -1
    W0[3 * n + 0][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U0[1 * m + 1][r] = 1
    V0[1 * p + 2][r] = 1
    V0[0 * p + 2][r] = -1
    W0[2 * n + 0][r] = 1
    W0[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U0[0 * m + 0][r] = 1
    U0[0 * m + 1][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = -1
    W0[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U0[1 * m + 0][r] = 1
    U0[0 * m + 0][r] = -1
    V0[0 * p + 2][r] = 1
    V0[0 * p + 3][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U0[0 * m + 1][r] = 1
    U0[1 * m + 1][r] = -1
    V0[1 * p + 2][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = 1
    r += 1
    
    return U0, V0, W0

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce the rank by identifying and removing linearly dependent terms.
    Uses QR with pivoting on the tensor matrix to find the numerical rank.
    """
    rank = len(U[0]) if U else 0
    dim = n * m * p
    
    # Build the tensor matrix M where M[idx, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    M = np.zeros((dim, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    M[idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # QR with column pivoting to find numerical rank
    try:
        from scipy.linalg import qr
        Q, R, piv = qr(M, pivoting=True, mode='economic')
    except Exception:
        Q, R = np.linalg.qr(M)
        piv = list(range(rank))
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = 0
    for r in range(rank):
        if abs(R[r, r]) > tol:
            num_rank = r + 1
        else:
            break
    
    if num_rank >= rank:
        return U, V, W, rank
    
    # num_rank < rank, so there are linear dependencies.
    # Try to remove terms one at a time from the end (least significant in QR pivoting)
    # and check if the remaining terms still span the tensor.
    
    # Get the pivot order
    pivot_order = list(piv)
    
    # Try removing terms in reverse pivot order (least important first)
    for remove_idx in range(rank - 1, num_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Build new U, V, W without this term
        keep_terms = [i for i in range(rank) if i != pivot_order[remove_idx]]
        new_rank = len(keep_terms)
        
        new_U = [[U[i][r] for r in keep_terms] for i in range(n * m)]
        new_V = [[V[i][r] for r in keep_terms] for i in range(m * p)]
        new_W = [[W[i][r] for r in keep_terms] for i in range(p * n)]
        
        # Verify the new decomposition
        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
            return new_U, new_V, new_W, new_rank
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the rank-28 decomposition
    U0, V0, W0 = _build_rank_28(n, m, p)
    
    # Verify the initial decomposition
    if not _verify_decomposition(U0, V0, W0, n, m, p):
        # Should not happen, but fallback
        return {"U": U0, "V": V0, "W": W0}
    
    # Try to reduce rank
    if time.time() - start_time < DEADLINE - 5:
        U, V, W, rank = _try_reduce_rank(U0, V0, W0, n, m, p, DEADLINE)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return {"U": U0, "V": V0, "W": W0}

# EVOLVE-BLOCK-END
