"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 135  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], np.dot(V[j*p+k], W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomp(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank_als(rank, n, m, p, max_iters=300, seed=42, time_limit=8):
    """
    Try to find a rank-rank decomposition using Accelerated Least Squares (ALS).
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    for it in range(max_iters):
        if time.time() > _START_TIME + time_limit:
            return None
            
        # Update U: for each (i,j), solve A @ u = ones(p) where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j] = u
                except:
                    return None
        
        # Update V: for each (j,k), solve A @ v = ones(n) where A[i,c] = U[i*m+j,c] * W[k*n+i,c]
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j * p + k] = v
                except:
                    return None
        
        # Update W: for each (k,i), solve A @ w = ones(m) where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k * n + i] = w
                except:
                    return None
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            if _verify(U, V, W, n, m, p, tol=1e-7):
                return U.tolist(), V.tolist(), W.tolist()
            
    # Final check
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return U.tolist(), V.tolist(), W.tolist()
        
    return None

def _try_rank_gradient(rank, n, m, p, max_iters=500, seed=42, time_limit=8):
    """
    Try to find a rank-rank decomposition using gradient descent with momentum.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Momentum
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    lr = 0.01
    momentum = 0.9
    
    for it in range(max_iters):
        if time.time() > _START_TIME + time_limit:
            return None
        
        # Compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j,c] of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                    prod_VW = V[j*p+k] * W[k*n+i]
                    gU[i*m+j] += 2 * (np.dot(U[i*m+j], prod_VW) - 1) * prod_VW
                    
                    prod_UW = U[i*m+j] * W[k*n+i]
                    gV[j*p+k] += 2 * (np.dot(V[j*p+k], prod_UW) - 1) * prod_UW
                    
                    prod_UV = U[i*m+j] * V[j*p+k]
                    gW[k*n+i] += 2 * (np.dot(W[k*n+i], prod_UV) - 1) * prod_UV
        
        # Update with momentum
        dU = momentum * dU + lr * gU
        dV = momentum * dV + lr * gV
        dW = momentum * dW + lr * gW
        
        U -= dU
        V -= dV
        W -= dW
        
        # Check convergence
        if it % 50 == 0:
            if _verify(U, V, W, n, m, p, tol=1e-7):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Decay learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.5
    
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return U.tolist(), V.tolist(), W.tolist()
        
    return None

def _try_rank_polish(rank, n, m, p, max_iters=50, seed=42, time_limit=5):
    """
    Try to polish a near-solution using alternating least squares with tight tolerance.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    for it in range(max_iters):
        if time.time() > _START_TIME + time_limit:
            return None
            
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j] = u
                except:
                    return None
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j * p + k] = v
                except:
                    return None
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k * n + i] = w
                except:
                    return None
        
        # Check convergence
        if it % 5 == 0:
            if _verify(U, V, W, n, m, p, tol=1e-10):
                return U.tolist(), V.tolist(), W.tolist()
            
    # Final check with very tight tolerance
    if _verify(U, V, W, n, m, p, tol=1e-10):
        return U.tolist(), V.tolist(), W.tolist()
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition as fallback
    U_fb, V_fb, W_fb = _schoolbook_decomp(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from 17 to 23
    # We'll try ALS first as it's more natural, then gradient if time permits
    
    # Prioritize lower ranks that are more likely to be achievable
    ranks_to_try = list(range(17, 24))
    
    for rank in ranks_to_try:
        if time.time() > _DEADLINE - 15:
            break
            
        found = False
        
        # Try ALS with multiple seeds
        for seed in range(10):
            if time.time() > _DEADLINE - 15:
                break
            result = _try_rank_als(rank, n, m, p, max_iters=200, seed=seed, time_limit=6)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
        
        # If ALS didn't work, try gradient descent with a few seeds
        if time.time() > _DEADLINE - 10:
            break
            
        for seed in range(3):
            if time.time() > _DEADLINE - 10:
                break
            result = _try_rank_gradient(rank, n, m, p, max_iters=300, seed=seed, time_limit=5)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
    
    # If we have time, try to polish the best solution found
    if time.time() < _DEADLINE - 20 and best_rank < schoolbook_rank:
        for rank in range(best_rank, max(best_rank - 2, 16)):
            if time.time() > _DEADLINE - 15:
                break
            for seed in range(5):
                if time.time() > _DEADLINE - 15:
                    break
                result = _try_rank_polish(rank, n, m, p, max_iters=30, seed=seed, time_limit=4)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p, tol=1e-10):
                        best_U, best_V, best_W = U, V, W
                        best_rank = rank
                        break
            if best_rank == rank:
                continue
            else:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
