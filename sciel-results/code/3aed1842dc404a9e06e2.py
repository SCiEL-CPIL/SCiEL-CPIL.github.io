import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct: T_recon[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.max(np.abs(T_recon - T)) < tol

def _als_step(U, V, W, T, n, m, p):
    """One ALS iteration: update U, V, W in turn."""
    rank = U.shape[1]
    # Update U: for each a, U[a,:] should minimize ||T[a,:,:] - sum_d U[a,d]*V[:,d]*W[:,d]||
    # T[a,:,:] = sum_d U[a,d] * (V[:,d] * W[:,d])  (outer product per d, summed)
    # So: T[a,:,:].reshape(m*p*p*n) = U[a,:] @ diag(V*W) ... 
    # Better: T[a,:,:] = sum_d U[a,d] * outer(V[:,d], W[:,d])
    # Flatten: vec(T[a,:,:]) = sum_d U[a,d] * vec(outer(V[:,d], W[:,d]))
    # = sum_d U[a,d] * (V[:,d] * W[:,d])  (element-wise, then flatten)
    # So the design matrix for U[a,:] is [V[:,0]*W[:,0], V[:,1]*W[:,1], ...] (each column is V[:,d]*W[:,d])
    
    # Precompute VW = V * W (element-wise), shape (rank, m*p) x ... no
    # V[:,d] has shape (m*p,), W[:,d] has shape (p*n,)
    # outer(V[:,d], W[:,d]) has shape (m*p, p*n)
    # vec of that is V[:,d] * W[:,d] (Kronecker-like, element-wise product flattened)
    
    # For updating U: 
    # T[a, :, :] = sum_d U[a,d] * outer(V[:,d], W[:,d])
    # Let's build the matrix M_U where M_U[:, d] = vec(outer(V[:,d], W[:,d])) = (V[:,d] * W[:,d]) reshaped
    # Actually V[:,d] * W[:,d] doesn't work directly since they have different sizes.
    # outer(V[:,d], W[:,d]).flatten() = np.kron(V[:,d], W[:,d])
    
    # This is expensive for large problems but our problem is small.
    
    # Update U
    for a in range(n * m):
        # T[a, :, :] shape (m*p, p*n)
        # We want: T[a,:,:] = sum_d U[a,d] * outer(V[:,d], W[:,d])
        # Reshape T[a,:,:] to (m*p*p*n,)
        target = T[a].flatten()
        # Design matrix: (m*p*p*n, rank), column d = kron(V[:,d], W[:,d])
        # This is (m*p)*(p*n) = 18*18 = 324 rows, rank columns
        # Build design matrix
        design = np.zeros((m * p * p * n, rank))
        for d in range(rank):
            design[:, d] = np.kron(V[:, d], W[:, d])
        # Solve: design @ U[a,:] = target
        sol, residuals, rank_sv, sv = np.linalg.lstsq(design, target, rcond=None)
        U[a, :] = sol
    
    # Update V
    for b in range(m * p):
        target = T[:, b, :].flatten()  # shape (n*m*p*n,)
        design = np.zeros((n * m * p * n, rank))
        for d in range(rank):
            design[:, d] = np.kron(U[:, d], W[:, d])
        sol, residuals, rank_sv, sv = np.linalg.lstsq(design, target, rcond=None)
        V[b, :] = sol
    
    # Update W
    for c in range(p * n):
        target = T[:, :, c].flatten()  # shape (n*m*m*p,)
        design = np.zeros((n * m * m * p, rank))
        for d in range(rank):
            design[:, d] = np.kron(U[:, d], V[:, d])
        sol, residuals, rank_sv, sv = np.linalg.lstsq(design, target, rcond=None)
        W[c, :] = sol
    
    return U, V, W

def _search_rank(T, max_rank, n, m, p, time_limit, n_restarts=5, n_iters=50):
    """Search for a decomposition of rank up to max_rank."""
    start_time = time.time()
    best_result = None
    
    for r in range(18, max_rank + 1):  # Start from theoretical lower bound
        if time.time() - start_time > time_limit * 0.8:
            break
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit * 0.9:
                break
            # Random initialization
            np.random.seed(42 + r * 1000 + restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(n_iters):
                if time.time() - start_time > time_limit * 0.95:
                    break
                U, V, W = _als_step(U, V, W, T, n, m, p)
                
                # Check convergence
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-9:
                    # Found exact decomposition
                    best_result = (r, U.copy(), V.copy(), W.copy())
                    return best_result
            
            # Check after all iterations
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-9:
                best_result = (r, U.copy(), V.copy(), W.copy())
                return best_result
    
    return best_result

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Try to find low-rank decomposition
    result = _search_rank(T, 54, n, m, p, time_limit)
    
    if result is not None:
        r, U, V, W = result
        # Convert to list format
        U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
