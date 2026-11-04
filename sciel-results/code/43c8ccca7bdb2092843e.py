import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, with 10s safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate_np(U, V, W, n, m, p):
    """
    Validate decomposition using NumPy.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Target tensor T[i, j, k] = 1 if indices match, else 0.
    Actually, the target is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k?
    No, the target is 1 for all i,j,k in the *linked* sense.
    Wait, in the flattened basis:
    The entry (i*m+j, j*p+k, k*n+i) must be 1.
    All other entries must be 0.
    
    Let's construct the target tensor T of shape (n*m, m*p, p*n).
    T is zero everywhere, except T[i*m+j, j*p+k, k*n+i] = 1.
    """
    R = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Compute the decomposition tensor D
    # D[i_idx, j_idx, k_idx] = sum_r U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
    # We can do this efficiently using einsum or loops if R is small.
    # n*m=16, m*p=20, p*n=20. R is small.
    
    # Initialize D
    D = np.zeros((n*m, m*p, p*n), dtype=int)
    
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Outer product
        D += np.einsum('i,j,k->ijk', u, v, w)
        
    # Construct Target Tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
                
    # Check if D == T
    if not np.array_equal(D, T):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def _random_search(n, m, p, target_rank, n_attempts=500):
    """
    Random search for a valid decomposition at the given rank.
    """
    for attempt in range(n_attempts):
        if _time_left() < 5:
            return None
        
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        # Density tuning: for small ranks, we need higher density
        # Heuristic: density ~ sqrt(1/R) or similar
        density = max(0.1, 1.0 / (target_rank ** 0.5))
        
        for r in range(target_rank):
            for i in range(n * m):
                if random.random() < density:
                    U[i, r] = 1 if random.random() < 0.5 else -1
            for j in range(m * p):
                if random.random() < density:
                    V[j, r] = 1 if random.random() < 0.5 else -1
            for k in range(p * n):
                if random.random() < density:
                    W[k, r] = 1 if random.random() < 0.5 else -1
        
        if _validate_np(U, V, W, n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = (U, V, W)
    best_rank = n * m * p
    
    # Try to find a lower rank
    lb = max(n * m, m * p, n * p)
    
    # Search from lb up to best_rank - 1
    for target_rank in range(lb, best_rank):
        if _time_left() < 10:
            break
            
        # Try random search
        result = _random_search(n, m, p, target_rank, n_attempts=100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # If we found a very low rank, we might stop early?
            # Or continue to see if we can go lower.
            # Given the difficulty, finding one is a big win.
            # Let's try a few more ranks if time permits.
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
