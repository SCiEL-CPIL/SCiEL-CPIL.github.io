"""Matrix multiplication 3x4x8 tensor decomposition search."""
import itertools
import random
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 96

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to C[i,k].
    Convention: W index is k*n+i.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # For each c, the term is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Sum over c should equal 1 for all (i,j,k)
    # Efficient check: build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # For each (i,j,k), sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use einsum
    # U_r[i,j,c], V_r[j,k,c], W_r[k,i,c] -> sum over c, result[i,j,k]
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _all_half_integers(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val, tol):
                    return False
    return True

def _random_search(start_rank, max_rank, num_trials, deadline):
    """Random search for a decomposition of given rank."""
    n, m, p = N, M, P
    best = None
    best_rank = max_rank + 1
    rng = random.Random(42)
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            # Generate random half-integer matrices
            scale = 2  # entries in {-1, 0, 1} * scale/2 = {-1, 0, 1}
            U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify_fast(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best = {"U": U, "V": V, "W": W}
                    return best
    return best

def _structured_search(deadline):
    """Try structured patterns that might yield lower rank."""
    n, m, p = N, M, P
    best = None
    best_rank = RANK_SCHOOLBOOK
    
    # Try rank 90-95 with some zero patterns
    for rank in range(90, 96):
        if time.time() > deadline:
            break
        # Start from schoolbook and try to merge columns
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Initialize with schoolbook-like pattern
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        if _verify_fast(U, V, W):
            if rank < best_rank:
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try structured search
    if time.time() < deadline:
        s = _structured_search(deadline)
        if s is not None:
            result = s
            best_rank = RANK_SCHOOLBOOK - 1  # at least one less
    
    # Try random search for lower ranks
    if time.time() < deadline:
        r = _random_search(85, 95, 2000, deadline)
        if r is not None:
            result = r
            best_rank = 85
    
    return result
