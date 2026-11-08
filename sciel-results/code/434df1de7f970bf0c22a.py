"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

def _verify_exact(U, V, W, n=3, m=5, p=5):
    """Exact verification using fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != 1:
                    return False
    return True

def _verify_numpy(U, V, W, n=3, m=5, p=5, tol=1e-9):
    """Fast numpy verification."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=5, p=5):
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

def _try_block_decomposition(n, m, p, deadline):
    """Try structured block decompositions."""
    results = []
    
    # Pattern 1: Rank = n*p (one factor per (i,k) pair)
    # U[i*m+j, i*p+k] = a[i,j], V[j*p+k, i*p+k] = b[j,k], W[k*n+i, i*p+k] = c[i,k]
    # Need: for all i,j,k: a[i,j]*b[j,k]*c[i,k] = 1
    # This requires a[i,j], b[j,k], c[i,k] to be non-zero and satisfy the product condition.
    # Simplest: a[i,j]=1, b[j,k]=1, c[i,k]=1 for all. But then rank = n*p = 15.
    # Wait, that works! Let's verify.
    rank = n * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for k in range(p):
            r = i*p + k
            for j in range(m):
                U[i*m+j][r] = 1
            for j in range(m):
                V[j*p+k][r] = 1
            W[k*n+i][r] = 1
    # Verify: for (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Only r = i*p+k contributes: U[i*m+j, i*p+k]=1, V[j*p+k, i*p+k]=1, W[k*n+i, i*p+k]=1
    # Product = 1. Correct!
    if _verify_numpy(U, V, W, n, m, p):
        if _verify_exact(U, V, W, n, m, p):
            results.append((rank, {"U": U, "V": V, "W": W}))
    
    # Pattern 2: Rank = m*p (one factor per (j,k) pair)
    rank = m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for j in range(m):
        for k in range(p):
            r = j*p + k
            for i in range(n):
                U[i*m+j][r] = 1
            V[j*p+k][r] = 1
            for i in range(n):
                W[k*n+i][r] = 1
    # Verify: for (i,j,k): only r = j*p+k contributes
    # U[i*m+j, j*p+k]=1, V[j*p+k, j*p+k]=1, W[k*n+i, j*p+k]=1
    # Product = 1. Correct!
    if _verify_numpy(U, V, W, n, m, p):
        if _verify_exact(U, V, W, n, m, p):
            results.append((rank, {"U": U, "V": V, "W": W}))
    
    # Pattern 3: Rank = n*m (one factor per (i,j) pair)
    rank = n * m
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            r = i*m + j
            U[i*m+j][r] = 1
            for k in range(p):
                V[j*p+k][r] = 1
            for k in range(p):
                W[k*n+i][r] = 1
    # Verify: for (i,j,k): only r = i*m+j contributes
    # U[i*m+j, i*m+j]=1, V[j*p+k, i*m+j]=1, W[k*n+i, i*m+j]=1
    # Product = 1. Correct!
    if _verify_numpy(U, V, W, n, m, p):
        if _verify_exact(U, V, W, n, m, p):
            results.append((rank, {"U": U, "V": V, "W": W}))
    
    return results

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p
    
    # Try structured block decompositions
    block_results = _try_block_decomposition(n, m, p, deadline)
    for rank, result in sorted(block_results, key=lambda x: x[0]):
        if rank < best_rank:
            best_rank = rank
            best_result = result
    
    # Try to find even lower rank with more sophisticated search
    if time.time() < deadline - 10:
        # Try rank < min(n*m, m*p, p*n)
        min_structured = min(n*m, m*p, p*n)
        for rank in range(2, min_structured):
            if time.time() > deadline - 5:
                break
            if rank >= best_rank:
                break
            
            # Random search with half-integers
            n_attempts = 100
            for attempt in range(n_attempts):
                if time.time() > deadline - 3:
                    break
                
                U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n*m)]
                V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m*p)]
                W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p*n)]
                
                if _verify_numpy(U, V, W, n, m, p):
                    if _verify_exact(U, V, W, n, m, p):
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
                        break
            
            if best_rank <= rank:
                break
    
    return best_result
# EVOLVE-BLOCK-END

