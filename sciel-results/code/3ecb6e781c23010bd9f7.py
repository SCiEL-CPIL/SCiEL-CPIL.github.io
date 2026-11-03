import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=500):
    """
    Try to reduce rank to target_rank using alternating minimization.
    We minimize the reconstruction error subject to rank constraint.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # We need to find a rank-target_rank decomposition
    # Strategy: randomly initialize and use alternating least squares
    np.random.seed(42)
    
    U_new = np.zeros((n*m, target_rank))
    V_new = np.zeros((m*p, target_rank))
    W_new = np.zeros((p*n, target_rank))
    
    # Initialize with random values scaled appropriately
    scale = 1.0 / (target_rank ** 0.5)
    U_new = np.random.randn(n*m, target_rank) * scale
    V_new = np.random.randn(m*p, target_rank) * scale
    W_new = np.random.randn(p*n, target_rank) * scale
    
    # Alternating minimization
    for it in range(max_iter):
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+0:, :] * W[0*n+i:, :]) = 1 (vector of ones)
                # But V and W have different k indices, so we need to be careful
                
                # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product)
                # Then: sum_k [sum_r U[i*m+j, r] * A_k[r]] = p? No, each equation is separate.
                
                # Actually, we have p equations:
                # For k=0: U[i*m+j, :] @ (V[j*p+0, :] * W[0*n+i, :]) = 1
                # For k=1: U[i*m+j, :] @ (V[j*p+1, :] * W[1*n+i, :]) = 1
                # etc.
                
                # Build the system: M @ u = b where M is p x target_rank
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                
                # Solve least squares
                if target_rank >= p:
                    u, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                else:
                    # Underdetermined: use pseudoinverse
                    u = np.linalg.pinv(M) @ b
                
                U_new[i*m+j, :] = u
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # For each i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A_i = U[i*m+j, :] * W[k*n+i, :]
                # Then: V[j*p+k, :] @ A_i = 1 for all i
                
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                
                if target_rank >= n:
                    v, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                else:
                    v = np.linalg.pinv(M) @ b
                
                V_new[j*p+k, :] = v
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A_j = U[i*m+j, :] * V[j*p+k, :]
                # Then: W[k*n+i, :] @ A_j = 1 for all j
                
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                b = np.ones(m)
                
                if target_rank >= m:
                    w, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                else:
                    w = np.linalg.pinv(M) @ b
                
                W_new[k*n+i, :] = w
        
        # Check convergence
        if it % 50 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        err += (val - 1.0) ** 2
            if err < 1e-12:
                break
    
    # Verify
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # 140s deadline (10s safety margin)
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to reduce rank incrementally
    # The lower bound is max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # But practical lower bounds are higher. Try from 99 down to 50
    
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline - 30:
            break
        
        U_try, V_try, W_try = _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=200)
        
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            best_rank = U_try.shape[1]
            U, V, W = best_U, best_V, best_W
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
