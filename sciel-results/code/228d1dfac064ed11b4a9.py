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
    Try to find a rank-r exact decomposition using structured initialization and local search.
    """
    n, m, p = 3, 5, 6
    N = n * m
    M = m * p
    P = p * n
    
    rng = np.random.RandomState(seed)
    start = time.time()
    
    # Structured initialization: use block-based assignment
    U = np.zeros((N, r))
    V = np.zeros((M, r))
    W = np.zeros((P, r))
    
    # Distribute the 90 schoolbook terms across r columns with structured overlap
    # Use a deterministic hash to assign each (i,j,k) triple to one or more columns
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Primary assignment
                col_primary = (i * m * p + j * p + k) % r
                U[i * m + j, col_primary] += 1.0
                V[j * p + k, col_primary] += 1.0
                W[k * n + i, col_primary] += 1.0
    
    # Local search with gradient-like updates
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    result = np.einsum('ijc,jkc,kic->ik', U2, V2, W2)
    best_err = np.sum((result - 1.0) ** 2)
    
    lr = 0.5
    for step in range(2000):
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
        
        if err < 1e-10:
            break
        
        # Compute gradients
        diff = result - 1.0  # shape (n, p)
        
        # dU[i,j,c] = sum_k diff[i,k] * V[j,k,c] * W[k,i,c]
        grad_U = np.zeros((n, m, r))
        grad_V = np.zeros((m, p, r))
        grad_W = np.zeros((p, n, r))
        
        for c in range(r):
            # grad_U[i,j,c] = sum_k diff[i,k] * V2[j,k,c] * W2[k,i,c]
            grad_U[:, :, c] = np.einsum('ik,jk,ki->ij', diff, V2[:, :, c], W2[:, :, c])
            # grad_V[j,k,c] = sum_i diff[i,k] * U2[i,j,c] * W2[k,i,c]
            grad_V[:, :, c] = np.einsum('ik,ij,ki->jk', diff, U2[:, :, c], W2[:, :, c])
            # grad_W[k,i,c] = sum_j diff[i,k] * U2[i,j,c] * V2[j,k,c]
            grad_W[:, :, c] = np.einsum('ik,ij,jk->ki', diff, U2[:, :, c], V2[:, :, c])
        
        # Adaptive learning rate
        err_prev = best_err
        U -= lr * grad_U.reshape(N, r)
        V -= lr * grad_V.reshape(M, r)
        W -= lr * grad_W.reshape(P, r)
        
        # If error increased, reduce learning rate
        U2 = U.reshape(n, m, r)
        V2 = V.reshape(m, p, r)
        W2 = W.reshape(p, n, r)
        result_new = np.einsum('ijc,jkc,kic->ik', U2, V2, W2)
        err_new = np.sum((result_new - 1.0) ** 2)
        
        if err_new > err * 1.1:
            # Revert and reduce lr
            U += lr * grad_U.reshape(N, r)
            V += lr * grad_V.reshape(M, r)
            W += lr * grad_W.reshape(P, r)
            lr *= 0.5
            if lr < 1e-6:
                break
        else:
            lr = min(lr * 1.05, 1.0)
    
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
    
    # Try to find lower-rank decompositions with structured search
    # Search from higher ranks down, trying multiple seeds per rank
    for r in range(85, 55, -1):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 2.0:
            break
        
        # Try multiple seeds for this rank
        n_attempts = max(1, int(remaining / 3.0))
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            seed = 1000 + r * 10 + attempt
            tlim = min(3.0, deadline - time.time() - 4)
            if tlim < 0.5:
                break
            
            try:
                result = _try_rank_r(r, seed=seed, time_limit=tlim)
                if result is not None:
                    U, V, W = result
                    best_rank = r
                    # Verify
                    if _verify(U, V, W, tol=1e-8):
                        break
            except Exception:
                continue
        
        if best_rank < 90:
            break
    
    # Final verification
    if not _verify(U, V, W, tol=1e-8):
        # Fall back to schoolbook if verification fails
        U, V, W = _schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
