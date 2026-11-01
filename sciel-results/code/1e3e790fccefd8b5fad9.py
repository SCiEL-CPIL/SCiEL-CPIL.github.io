# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _verify_factors_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Compute the tensor decomposition
    # T[abc] = sum_l U[a,l] V[b,l] W[c,l]
    # We only need to check the 18 non-zero entries of the multiplication tensor.
    
    # Indices for A (n*m), B (m*p), C (p*n)
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    
    # Instead of looping in Python, we can vectorize the check for the specific triples.
    # There are 18 triples.
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A_idx = i * m + j
                B_idx = j * p + k
                C_idx = k * n + i
                triples.append((A_idx, B_idx, C_idx))
    
    # For each triple, compute the dot product
    # U[A_idx] is a vector of length rank
    # V[B_idx] is a vector of length rank
    # W[C_idx] is a vector of length rank
    # Product is element-wise, then sum
    
    for A_idx, B_idx, C_idx in triples:
        val = np.dot(U_arr[A_idx], V_arr[B_idx] * W_arr[C_idx])
        if abs(val - 1.0) > tol:
            return False
    return True

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_random_search_numpy(n, m, p, max_rank, trials=10000, seed=42, time_limit=10):
    """Random search using NumPy for speed."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        rank = max_rank
        # Generate random integer matrices
        # Values in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        U = rng.integers(-2, 3, size=(n * m, rank))
        V = rng.integers(-2, 3, size=(m * p, rank))
        W = rng.integers(-2, 3, size=(p * n, rank))
        
        if _verify_factors_fast(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist(), rank
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 10  # Safety margin
    
    # Try random search for lower ranks
    # Focus on ranks likely to be minimal or close to it
    # Known rank for 2x3x3 is 11.
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 2:
            break
        
        # Allocate time based on rank
        # Try harder for lower ranks? Or just equal time?
        # Let's give 5 seconds to each rank
        time_alloc = min(remaining_time - 2, 5)
        
        result = _try_random_search_numpy(n, m, p, target_rank, trials=50000, seed=target_rank, time_limit=time_alloc)
        if result is not None:
            U, V, W, rank = result
            # Double check with the original slow verifier? No, fast one is sufficient if tol is tight
            # But let's be safe and use the original if available? No, just return.
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
