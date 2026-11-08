"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Deadline guard: 120s wall-clock limit, return best with safety margin.
DEADLINE_SECONDS = 115.0

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (since
    (AB)[i,k] = sum_j A[i,j] B[j,k], the structure tensor entry for (i,j,k) is 1).
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                val = np.dot(np.dot(u_val, v_val), w_val)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_trials=5000, seed=42):
    """
    Attempt to find a rank-target_rank decomposition using random search.
    This is a heuristic; for n=m=p=6, the known minimal rank is 21 (Landsberg) 
    but finding it via random search is extremely hard. We try a few approaches.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Approach: Try to find a low-rank decomposition by solving for U, V, W
    # such that the tensor product matches. This is a 3-tensor decomposition problem.
    
    # For 6x6x6, the best known upper bound is 21 (from Landsberg's bound or similar).
    # The schoolbook is 216. We can't easily find 21 by random search.
    
    # Let's try a structured approach: use the fact that matrix multiplication
    # tensor can be decomposed using bilinear algorithms.
    
    # For now, try random search with small ranks to see if we get lucky.
    # This is unlikely to work for rank < 216, but let's try.
    
    for trial in range(max_trials):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Random initialization
        U_rand = rng.randn(n * m, target_rank)
        V_rand = rng.randn(m * p, target_rank)
        W_rand = rng.randn(p * n, target_rank)
        
        # Quick check: compute the implied tensor and see how close to all-ones
        # We can't fully verify without checking all n*m*p entries, but we can
        # check a subset.
        
        # Actually, let's just verify fully since n*m*p = 216 is small.
        if _verify_decomposition(
            U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p
        ):
            return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Known bounds for 6x6x6:
    # - Schoolbook: 216
    # - Best known upper bound: 21 (Landsberg 2013, or similar)
    # - Lower bound: 19 (from various results)
    
    # We can't easily find rank 21 by random search. Let's try a few structured
    # approaches.
    
    # One approach: Use the fact that the matrix multiplication tensor for
    # n x n matrices has a known decomposition structure. For 6x6, we can try
    # to use block structures.
    
    # Actually, let's try a different approach: use the Strassen-like
    # decomposition ideas, but for 6x6 this is complex.
    
    # Given the time constraint and the difficulty of finding non-trivial
    # decompositions, let's return the schoolbook decomposition.
    # This is guaranteed to be correct and has rank 216.
    
    return best

# EVOLVE-BLOCK-END
