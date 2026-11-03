"""Improved program: vectorized optimization for M_{4,5,6} rank minimization."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i*m + j]
                v = V[j*p + k]
                w = W[k*n + i]
                val = np.sum(u * v * w)
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    return max_err <= tol, max_err

def _compute_residuals(U, V, W, n, m, p):
    """Compute residual tensor R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1.
    Returns array of shape (n, m, p)."""
    R = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1.0
    return R

def _optimize_rank(U, V, W, n, m, p, max_iters=2000, lr=0.01, timeout=30, tol=1e-10):
    """Optimize U, V, W to minimize sum of squared residuals.
    Returns (success, U, V, W, final_error)."""
    start = time.time()
    nrm = n*m*p
    rank = U.shape[1]
    
    # Precompute index mappings for efficiency
    # For each (i,j,k), we need indices:
    # U_idx = i*m + j, V_idx = j*p + k, W_idx = k*n + i
    
    # Use a simple gradient descent with adaptive learning rate
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Compute residuals
        R = _compute_residuals(U, V, W, n, m, p)
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            return True, U, V, W, err
        
        # Compute gradients using vectorized operations
        # grad_U[i*m+j, c] = 2 * sum_k R[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # grad_V[j*p+k, c] = 2 * sum_i R[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
        # grad_W[k*n+i, c] = 2 * sum_j R[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = 2.0 * R[i, j, k]
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    # grad_U[u_idx, :] += r * V[v_idx, :] * W[w_idx, :]
                    grad_U[u_idx, :] += r * V[v_idx, :] * W[w_idx, :]
                    # grad_V[v_idx, :] += r * U[u_idx, :] * W[w_idx, :]
                    grad_V[v_idx, :] += r * U[u_idx, :] * W[w_idx, :]
                    # grad_W[w_idx, :] += r * U[u_idx, :] * V[v_idx, :]
                    grad_W[w_idx, :] += r * U[u_idx, :] * V[v_idx, :]
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate: reduce if error increases
        if err > best_err * 1.1:
            lr = max(lr * 0.5, 1e-8)
    
    # Final check
    final_err = _validate(best_U, best_V, best_W, n, m, p, tol=tol)[1]
    if final_err < tol:
        return True, best_U, best_V, best_W, final_err
    return False, best_U, best_V, best_W, final_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    # Try to improve: search for lower rank decompositions
    # Start from rank 119 down to 80
    for target_rank in range(119, max(79, 0), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(8):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(12345 + attempt * 7)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Also try a structured initialization: start from schoolbook and project
            if attempt == 0:
                # Use schoolbook columns as initial guess (first target_rank columns)
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
            
            success, U, V, W, err = _optimize_rank(U, V, W, n, m, p, 
                                                     max_iters=3000, lr=0.005, 
                                                     timeout=12, tol=1e-10)
            
            if success:
                valid, final_err = _validate(U, V, W, n, m, p, tol=1e-10)
                if valid:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # Can't improve further with this approach
            break
    
    # Convert to lists of lists as required
    U_list = [[float(x) for x in row] for row in np.asarray(best_U)]
    V_list = [[float(x) for x in row] for row in np.asarray(best_V)]
    W_list = [[float(x) for x in row] for row in np.asarray(best_W)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
