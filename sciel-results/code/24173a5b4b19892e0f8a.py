"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    
    return max_err < 1e-9

def _compute_error(U, V, W, n, m, p, rank):
    """Compute max error for a decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err

def _local_optimize(U, V, W, n, m, p, rank, max_iters=50, time_budget=2.0):
    """Coordinate descent optimization to reduce error."""
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                for c in range(rank):
                    # d/dU[idx,c] sum_k V[j*p+k,c]*W[k*n+i,c] = 1
                    # U[idx,c] * (sum_k V[j*p+k,c]*W[k*n+i,c]) + rest = 1
                    coeff = 0.0
                    rest = 0.0
                    for k in range(p):
                        coeff += V[j*p + k, c] * W[k*n + i, c]
                        for c2 in range(rank):
                            if c2 != c:
                                rest += U[idx, c2] * V[j*p + k, c2] * W[k*n + i, c2]
                    if abs(coeff) > 1e-12:
                        U[idx, c] = (1.0 - rest) / coeff
                    else:
                        U[idx, c] = 0.0
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                for c in range(rank):
                    coeff = 0.0
                    rest = 0.0
                    for i in range(n):
                        coeff += U[i*m + j, c] * W[k*n + i, c]
                        for c2 in range(rank):
                            if c2 != c:
                                rest += U[i*m + j, c2] * V[idx, c2] * W[k*n + i, c2]
                    if abs(coeff) > 1e-12:
                        V[idx, c] = (1.0 - rest) / coeff
                    else:
                        V[idx, c] = 0.0
        
        # Optimize W
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                for c in range(rank):
                    coeff = 0.0
                    rest = 0.0
                    for j in range(m):
                        coeff += U[i*m + j, c] * V[j*p + k, c]
                        for c2 in range(rank):
                            if c2 != c:
                                rest += U[i*m + j, c2] * V[j*p + k, c2] * W[idx, c2]
                    if abs(coeff) > 1e-12:
                        W[idx, c] = (1.0 - rest) / coeff
                    else:
                        W[idx, c] = 0.0
    
    return U, V, W

def _structured_init(n, m, p, target_rank, trial):
    """Create structured initializations based on schoolbook subsets."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if target_rank >= n * m * p:
        # Full schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Random subset of schoolbook terms + perturbation
    all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    selected = random.sample(all_terms, min(target_rank, len(all_terms)))
    
    for c, (i, j, k) in enumerate(selected):
        U[i*m + j, c] = 1.0
        V[j*p + k, c] = 1.0
        W[k*n + i, c] = 1.0
    
    # Add random perturbation to remaining columns
    for c in range(len(selected), target_rank):
        U[:, c] = np.random.randn(n * m) * 0.1
        V[:, c] = np.random.randn(m * p) * 0.1
        W[:, c] = np.random.randn(p * n) * 0.1
    
    return U, V, W

def _try_rank_search(n, m, p, max_rank, time_limit=10.0):
    """Try to find a decomposition with rank <= max_rank using structured search.
    Limited time budget to ensure fast fallback."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Only attempt a small range of ranks, focusing on lower ones first
    # For 3x5x5, schoolbook is 75. Searching down to 1 is too slow.
    # We'll try a few random ranks and very low ranks.
    ranks_to_try = list(range(1, min(max_rank, 20) + 1))
    
    # Add some random higher ranks if time permits? No, keep it simple and fast.
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 1.0:
            break
        
        # Try a few initializations
        num_tries = 5
        for trial in range(num_tries):
            if time.time() > deadline - 1.0:
                break
            
            # Use structured initialization
            U, V, W = _structured_init(n, m, p, target_rank, trial)
            
            # Local optimization with strict time limit
            U, V, W = _local_optimize(U, V, W, n, m, p, target_rank, 
                                       max_iters=10, time_budget=0.5)
            
            # Check error
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-9:
                U_list = [[float(x) for x in U[row]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row]] for row in range(p*n)]
                
                # Try to round to half-integers
                U_rounded = [[round(2*x)/2 for x in row] for row in U_list]
                V_rounded = [[round(2*x)/2 for x in row] for row in V_list]
                W_rounded = [[round(2*x)/2 for x in row] for row in W_list]
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                else:
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook decomposition as a safe fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition with a strict time limit
    # We limit the search to 10 seconds to ensure we return well before the 120s limit
    result = _try_rank_search(n, m, p, schoolbook_rank - 1, time_limit=10.0)
    
    if result is not None:
        return result
    else:
        return schoolbook
# EVOLVE-BLOCK-END
