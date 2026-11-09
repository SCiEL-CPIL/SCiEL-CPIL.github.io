import numpy as np
import time

def _verify(U, V, W, tol=1e-9):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    n, m, p = 3, 5, 6
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    U2 = U.reshape(n, m, rank)
    V2 = V.reshape(m, p, rank)
    W2 = W.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijc,jkc,kic->ik', U2, V2, W2)
    except Exception:
        return False
    
    if not np.allclose(result, 1.0, atol=tol, rtol=0):
        return False
    return True

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

def _try_rank_r(r, seed=42, time_limit=8.0):
    """
    Try to find a rank-r exact decomposition using gradient-based local search.
    """
    n, m, p = 3, 5, 6
    N = n * m
    M = m * p
    P = p * n
    
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Initialize with schoolbook-like structure but compressed
    U = np.zeros((N, r))
    V = np.zeros((M, r))
    W = np.zeros((P, r))
    
    # Use a structured initialization: distribute the 90 schoolbook terms into r columns
    # by averaging or using random projections
    for c in range(r):
        # Each column c covers a subset of the (i,j,k) triples
        # Simple approach: random assignment
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Assign this (i,j,k) to column (i*m*p + j*p + k) % r
                    col = (i * m * p + j * p + k) % r
                    U[i * m + j, col] += 1.0 / (r / (n * m * p))
                    V[j * p + k, col] += 1.0 / (r / (n * m * p))
                    W[k * n + i, col] += 1.0 / (r / (n * m * p))
    
    # Normalize
    for c in range(r):
        U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-10)
        V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-10)
        W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-10)
    
    # Local search: try small perturbations
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(500):
        if time.time() - start > time_limit:
            break
        
        # Compute current error
        U2 = U.reshape(n, m, r)
        V2 = V.reshape(m, p, r)
        W2 = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ik', U2, V2, W2)
        err = np.sum((result - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        # Random perturbation
        lr = 0.01
        U += lr * rng.randn(N, r) * 0.1
        V += lr * rng.randn(M, r) * 0.1
        W += lr * rng.randn(P, r) * 0.1
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    
    return None

def construct():
    """
    Returns the schoolbook tensor decomposition for 3x5 * 5x6 matrix multiplication,
    or a lower-rank decomposition if found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = 90
    
    # Try to find lower-rank decompositions
    for r in range(50, 90):
        if time.time() > deadline - 5:
            break
        
        try:
            result = _try_rank_r(r, seed=200 + r, time_limit=min(4.0, deadline - time.time() - 4))
            if result is not None:
                U, V, W = result
                best_rank = r
                break
        except Exception:
            continue
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
