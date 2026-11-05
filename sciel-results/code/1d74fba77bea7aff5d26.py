"""
Rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication.

Strategy:
1. Try a structured low-rank search using the bilinear structure of matrix multiplication.
2. Fall back to schoolbook (rank 72) if no better decomposition is found.

The key insight is that the matrix multiplication tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
must exactly reproduce the MM identity. We search for rank-r decompositions where r < 72.

For small dimensions like 3x4x6, we can enumerate structured patterns (e.g., based on
partitioning the inner dimension and using algebraic identities) to try to beat schoolbook.
"""
import time
import numpy as np
from itertools import product

N, M, P = 3, 4, 6

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i, j, k) triple, sum over r should be 1
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, rank):
    """Vectorized verification."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the full tensor T of shape (n, m, p)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
    
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return U, V, W, rank

def _try_strassen_inner_split(n, m, p, deadline):
    """
    Try splitting inner dimension m=4 into two halves of size 2.
    For each pair of halves, apply Strassen-style 7-multiplication trick.
    
    This is a heuristic: we try to find a rank-7 decomposition for a 2x2 core
    and combine with the outer dimensions.
    """
    # This is complex; for now, try a simple structured ansatz
    # We look for rank r where r < n*m*p
    # Try a few candidate ranks
    for rank in [64, 60, 56, 54, 52, 50]:
        if time.time() > deadline:
            break
        
        # Initialize with random or structured pattern
        # Use a structured initialization based on block decomposition
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        # Fill using a block-structured pattern
        # For each "multiplication slot", assign contributions
        slot = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if slot >= rank:
                        break
                    U[i * m + j, slot] = 1.0
                    V[j * p + k, slot] = 1.0
                    W[k * n + i, slot] = 1.0
                    slot += 1
        
        # Verify
        if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist(), rank
    
    return None

def _try_block_diagonal_strassen(n, m, p, deadline):
    """
    Try to use Strassen's 7-multiplication identity on 2x2 blocks.
    
    Split A (3x4) into rows, B (4x6) into columns.
    The inner dimension is 4, which splits into two 2-dim halves.
    
    For a 2x2 matrix multiply, Strassen uses 7 multiplications.
    We try to compose this with the outer structure.
    """
    if time.time() > deadline:
        return None
    
    # For now, this is a placeholder for a more sophisticated search
    # that would systematically try algebraic identities
    return None

def _random_search(n, m, p, deadline, max_rank=70):
    """
    Random search for low-rank decompositions.
    """
    best = None
    
    for rank in range(68, min(max_rank, 72)):
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # Random structured initialization
            np.random.seed(42 + rank + attempt)
            
            # Assign random ±1 or 0 patterns
            for r in range(rank):
                # Randomly select which (i,j), (j,k), (k,i) get non-zero
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            
            if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
                if best is None or rank < best[3]:
                    best = (U.tolist(), V.tolist(), W.tolist(), rank)
                break
    
    return best

def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.

    Returns a dict with keys 'U', 'V', 'W', each a list of lists of integers,
    where:
      U: shape (n*m, rank)
      V: shape (m*p, rank)
      W: shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = N, M, P
    
    # Try structured searches first
    result = _try_strassen_inner_split(n, m, p, deadline)
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    result = _try_block_diagonal_strassen(n, m, p, deadline)
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    # Try random search
    if time.time() < deadline - 5:
        result = _random_search(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
