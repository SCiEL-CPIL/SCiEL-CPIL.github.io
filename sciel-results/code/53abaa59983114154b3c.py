"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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


def _residual_sq(U, V, W, n, m, p):
    """Compute sum of squared residuals over all (i,k) entries.
    Residual[i,k] = sum_r U[i*m+? ... ] 
    Actually: C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i]
    We need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 if j is the middle index.
    Wait, the tensor is T[i,j,k] = delta... no.
    
    The matrix multiplication tensor: T[i,j,k] = 1 for all valid (i,j,k) where
    C[i,k] = sum_j A[i,j] B[j,k].
    
    So the tensor entry at (i,j,k) is 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    A rank-r decomposition gives:
    sum_{r=1}^{R} U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all (i,j,k)
    
    Residual = sum_{i,j,k} (sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] - 1)^2
    """
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                total += (s - 1.0) ** 2
    return total


def _try_rank(n, m, p, rank, seed, time_budget):
    """Try to find a rank-`rank` decomposition using random restarts + local search.
    Returns (U, V, W) if found with residual < 1e-10, else None.
    """
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    best_residual = float('inf')
    best_params = None
    
    num_params = rank * (n * m + m * p + p * n)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        params = [rng.gauss(0, 1.0) for _ in range(num_params)]
        
        # Local search: coordinate descent with random perturbations
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Evaluate current residual
            # Build U, V, W from params
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            idx = 0
            for i in range(n * m):
                for r in range(rank):
                    U[i][r] = params[idx]
                    idx += 1
            for j in range(m * p):
                for r in range(rank):
                    V[j][r] = params[idx]
                    idx += 1
            for k in range(p * n):
                for r in range(rank):
                    W[k][r] = params[idx]
                    idx += 1
            
            res = _residual_sq(U, V, W, n, m, p)
            
            if res < best_residual:
                best_residual = res
                best_params = params[:]
            
            if res < 1e-10:
                return U, V, W
            
            # Coordinate descent: try perturbing one coordinate
            if it % 10 == 0:
                # Try a random coordinate
                coord = rng.randint(0, num_params - 1)
                old_val = params[coord]
                delta = rng.gauss(0, 0.1)
                params[coord] = old_val + delta
                
                # Quick check: rebuild and evaluate (expensive but necessary)
                U2 = [[0] * rank for _ in range(n * m)]
                V2 = [[0] * rank for _ in range(m * p)]
                W2 = [[0] * rank for _ in range(p * n)]
                idx2 = 0
                for i in range(n * m):
                    for r in range(rank):
                        U2[i][r] = params[idx2]
                        idx2 += 1
                for j in range(m * p):
                    for r in range(rank):
                        V2[j][r] = params[idx2]
                        idx2 += 1
                for k in range(p * n):
                    for r in range(rank):
                        W2[k][r] = params[idx2]
                        idx2 += 1
                
                res2 = _residual_sq(U2, V2, W2, n, m, p)
                if res2 < res:
                    res = res2
                else:
                    params[coord] = old_val
        
        # After local search, check best
        if best_residual < 1e-10:
            # Rebuild from best_params
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            idx = 0
            for i in range(n * m):
                for r in range(rank):
                    U[i][r] = best_params[idx]
                    idx += 1
            for j in range(m * p):
                for r in range(rank):
                    V[j][r] = best_params[idx]
                    idx += 1
            for k in range(p * n):
                for r in range(rank):
                    W[k][r] = best_params[idx]
                    idx += 1
            return U, V, W
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses random restarts with local search, trying ranks from low to high.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try ranks from minimum possible up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Known upper bounds suggest we can do better than 36
    
    for rank in range(12, 37):
        if time.time() > deadline:
            break
        
        time_budget = min(3.0, deadline - time.time())
        if time_budget < 0.5:
            break
        
        result = _try_rank(n, m, p, rank, seed=rank * 12345, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            # Snap to nearest rational if close
            # For now, return as-is
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
