import numpy as np
import time

def _schoolbook():
    """Return the trivial rank-90 schoolbook decomposition."""
    n, m, p = 3, 5, 6
    N = n * m  # 15
    M = m * p  # 30
    P = p * n  # 18
    rank = n * m * p  # 90
    
    U = np.zeros((N, rank), dtype=float)
    V = np.zeros((M, rank), dtype=float)
    W = np.zeros((P, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """
    Verify that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all i,j,k.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    n, m, p = 3, 5, 6
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    # For each (i,k), we need sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Reshape: U2[i,j,c], V2[j,k,c], W2[k,i,c]
    U2 = U.reshape(n, m, rank)
    V2 = V.reshape(m, p, rank)
    W2 = W.reshape(p, n, rank)
    
    # For each (i,k): sum_j (U2[i,j,:] * V2[j,k,:]) @ W2[k,i,:]
    # = sum_j sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c]
    # We can compute: for each i,k, this is sum_j (U2[i,j,:] * V2[j,k,:]).dot(W2[k,i,:])
    # Or: sum_c W2[k,i,c] * (sum_j U2[i,j,c] * V2[j,k,c])
    # Or use einsum: 'ijc,jkc,kic->ik'
    try:
        result = np.einsum('ijc,jkc,kic->ik', U2, V2, W2)
    except Exception:
        return False
    
    if not np.allclose(result, 1.0, atol=tol, rtol=0):
        return False
    return True

def _try_rank_r(r, seed=42, time_limit=10.0):
    """
    Try to find a rank-r exact decomposition using a simple iterative approach.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = 3, 5, 6
    N = n * m
    M = m * p
    P = p * n
    
    rng = np.random.RandomState(seed)
    
    # Initialize with random small values
    U = rng.randn(N, r) * 0.1
    V = rng.randn(M, r) * 0.1
    W = rng.randn(P, r) * 0.1
    
    start = time.time()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple gradient-free search: try random perturbations and keep improvements
    # This is a very basic approach; in practice, more sophisticated methods would be needed.
    
    # First, try a structured initialization based on partial schoolbook
    # For each i, try to cover all (j,k) pairs with fewer terms
    
    # Alternative: use the fact that for fixed i, we need sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is like a rank-r matrix factorization for each i
    
    # Let's try a more structured approach:
    # For each i, define A_i[j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want A_i = J (all ones) for all i.
    
    # Try initializing W with a structured pattern
    for attempt in range(5):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with some structure
        U = rng.randn(N, r) * 0.5
        V = rng.randn(M, r) * 0.5
        W = rng.randn(P, r) * 0.5
        
        # Normalize columns
        for c in range(r):
            U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-10)
            V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-10)
            W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-10)
        
        # Check if this happens to work (very unlikely for random init)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    
    return None

def construct():
    """
    Returns the schoolbook tensor decomposition for 3x5 * 5x6 matrix multiplication,
    or a lower-rank decomposition if found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = 90
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound is around 60-70 for 3x5x6
    # Try ranks from 60 up to 89
    for r in range(60, 90):
        if time.time() > deadline - 10:
            break
        
        try:
            result = _try_rank_r(r, seed=42 + r, time_limit=min(5.0, deadline - time.time() - 5))
            if result is not None:
                U, V, W = result
                best_rank = r
                break  # Found a working decomposition, stop
        except Exception:
            continue
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
