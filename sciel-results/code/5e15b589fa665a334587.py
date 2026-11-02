"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Actually, the convention is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor M[i,j,k] = delta_{j,j} = 1 for all valid i,j,k (since A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # The standard matrix multiplication tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, T[i,j,k] = 1 for all i,j,k because every pair (A[i,j], B[j,k]) contributes to C[i,k].
    # No wait. The tensor is defined such that C[i,k] = sum_{j} A[i,j] * B[j,k].
    # The decomposition: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) ... no that's not right either.
    
    # Let me think more carefully. The standard formulation:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition gives: C[i,k] = sum_c sum_j U[i*m+j, c] A[i,j] * V[j*p+k, c] B[j,k]
    # Wait, that doesn't work dimensionally. Let me re-read.
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The convention is W[k*n+i, r]
    
    # So the decomposition is:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] ... no, that's not right.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # The multiplication tensor T is a 3-tensor with T[i,j,k] = 1 if i,j,k form a valid "chain"
    # i.e., A[i,j] * B[j,k] contributes to C[i,k].
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # A rank-r decomposition of T means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where the indices are flattened as specified.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=42):
    """Generate a random rank-r decomposition and try to verify it."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    return U, V, W

def _try_rank(n, m, p, rank, time_limit, seed_base=0):
    """Try to find a valid rank-r decomposition within time_limit seconds."""
    start_time = time.time()
    
    # For small ranks, try random initializations with verification
    # This is unlikely to work for large gaps, but let's try.
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            return None
        
        seed = seed_base + attempt
        U, V, W = _random_decomposition(n, m, p, rank, seed=seed)
        
        # Quick check: verify a few random (i,j,k) tuples
        rng = np.random.default_rng(seed + 10000)
        valid = True
        for _ in range(20):
            i = rng.integers(0, n)
            j = rng.integers(0, m)
            k = rng.integers(0, p)
            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(s - 1.0) > 1e-6:
                valid = False
                break
        if valid:
            # Full verification
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook (rank 100)
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Known lower bounds: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # Upper bound: n*m*p = 100
    # Try ranks from 25 upward
    
    lower_bound = max(n*m, m*p, p*n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:  # Leave 30s for final verification
            break
        
        time_for_this_rank = min(15, deadline - time.time() - 30)
        if time_for_this_rank <= 0:
            break
        
        result = _try_rank(n, m, p, rank, time_for_this_rank, seed_base=rank * 1000)
        if result is not None:
            best = result
            best_rank = rank
            # Found a valid decomposition, continue trying lower? No, we're going up.
            # Actually we want the minimum, so if we find one at this rank, we can stop
            # only if this is the lower bound. Otherwise continue.
            break
    
    return best
# EVOLVE-BLOCK-END
