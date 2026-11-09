"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _compute_error_vectorized(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    r = U_np.shape[1]
    
    # Build index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = ii * m + jj
    idx_v = jj * p + kk
    idx_w = kk * n + ii
    
    # For each (i,j,k), compute sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
    # U[idx_u, :] has shape (n*m*p, r)
    U_sel = U_np[idx_u.flatten(), :]  # (n*m*p, r)
    V_sel = V_np[idx_v.flatten(), :]  # (n*m*p, r)
    W_sel = W_np[idx_w.flatten(), :]  # (n*m*p, r)
    
    # Element-wise product then sum over c
    prod = U_sel * V_sel * W_sel  # (n*m*p, r)
    vals = prod.sum(axis=1)  # (n*m*p,)
    
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _als_step_U(U, V, W, idx_u, idx_v, idx_w, n, m, p):
    """ALS step to update U given fixed V and W."""
    r = U.shape[1]
    N = len(idx_u)
    
    V_sel = V[idx_v, :]  # (N, r)
    W_sel = W[idx_w, :]  # (N, r)
    VW = V_sel * W_sel  # (N, r)
    
    U_new = np.zeros_like(U)
    for ui in range(n * m):
        mask = idx_u == ui
        if not np.any(mask):
            U_new[ui, :] = 0
            continue
        
        VW_sub = VW[mask, :]  # (num_t, r)
        # Solve VW_sub @ U[ui, :] = ones(num_t)
        sol, _, _, _ = np.linalg.lstsq(VW_sub, np.ones(len(VW_sub)), rcond=None)
        U_new[ui, :] = sol
    
    return U_new

def _als_step_V(U, V, W, idx_u, idx_v, idx_w, n, m, p):
    """ALS step to update V given fixed U and W."""
    r = V.shape[1]
    N = len(idx_v)
    
    U_sel = U[idx_u, :]  # (N, r)
    W_sel = W[idx_w, :]  # (N, r)
    UW = U_sel * W_sel  # (N, r)
    
    V_new = np.zeros_like(V)
    for vi in range(m * p):
        mask = idx_v == vi
        if not np.any(mask):
            V_new[vi, :] = 0
            continue
        
        UW_sub = UW[mask, :]  # (num_t, r)
        sol, _, _, _ = np.linalg.lstsq(UW_sub, np.ones(len(UW_sub)), rcond=None)
        V_new[vi, :] = sol
    
    return V_new

def _als_step_W(U, V, W, idx_u, idx_v, idx_w, n, m, p):
    """ALS step to update W given fixed U and V."""
    r = W.shape[1]
    N = len(idx_w)
    
    U_sel = U[idx_u, :]  # (N, r)
    V_sel = V[idx_v, :]  # (N, r)
    UV = U_sel * V_sel  # (N, r)
    
    W_new = np.zeros_like(W)
    for wi in range(p * n):
        mask = idx_w == wi
        if not np.any(mask):
            W_new[wi, :] = 0
            continue
        
        UV_sub = UV[mask, :]  # (num_t, r)
        sol, _, _, _ = np.linalg.lstsq(UV_sub, np.ones(len(UV_sub)), rcond=None)
        W_new[wi, :] = sol
    
    return W_new

def _als_decomposition(target_rank, num_attempts=10, max_iter=200, deadline=None):
    """Try to find a rank decomposition using ALS with better initialization."""
    n, m, p = DIMENSIONS
    
    # Precompute index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()  # (n*m*p,)
    idx_v = (jj * p + kk).flatten()  # (n*m*p,)
    idx_w = (kk * n + ii).flatten()  # (n*m*p,)
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 10:
            return None
            
        np.random.seed(attempt * 104729 + target_rank * 31)
        
        # Sparse initialization often helps
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Zero out some entries to encourage sparsity
        mask_u = np.random.rand(n*m, target_rank) > 0.5
        mask_v = np.random.rand(m*p, target_rank) > 0.5
        mask_w = np.random.rand(p*n, target_rank) > 0.5
        U[mask_u] = 0
        V[mask_v] = 0
        W[mask_w] = 0
        
        # Scale to prevent explosion
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 10:
                break
            
            # ALS updates
            U = _als_step_U(U, V, W, idx_u, idx_v, idx_w, n, m, p)
            V = _als_step_V(U, V, W, idx_u, idx_v, idx_w, n, m, p)
            W = _als_step_W(U, V, W, idx_u, idx_v, idx_w, n, m, p)
            
            # Check error
            err = _compute_error_vectorized(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-6:
            # Try to round to integers/half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error_vectorized(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-6:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None

def _try_rank_decomposition(target_rank, num_attempts=10, max_iter=500, deadline=None):
    """Try to find a rank decomposition using gradient descent with restarts."""
    n, m, p = DIMENSIONS
    
    # Precompute index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()  # (n*m*p,)
    idx_v = (jj * p + kk).flatten()  # (n*m*p,)
    idx_w = (kk * n + ii).flatten()  # (n*m*p,)
    N = n * m * p  # 120
    
    # First try ALS
    result = _als_decomposition(target_rank, num_attempts=num_attempts, max_iter=200, deadline=deadline)
    if result is not None:
        return result
    
    # Fall back to gradient descent with restarts
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 10:
            return None
            
        np.random.seed(attempt * 7919 + target_rank)
        
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 10:
                break
            
            # Compute residuals
            U_sel = U[idx_u, :]  # (N, r)
            V_sel = V[idx_v, :]  # (N, r)
            W_sel = W[idx_w, :]  # (N, r)
            
            prod = U_sel * V_sel * W_sel  # (N, r)
            vals = prod.sum(axis=1)  # (N,)
            residuals = vals - 1.0  # (N,)
            
            # Gradients
            VW = V_sel * W_sel  # (N, r)
            UW = U_sel * W_sel  # (N, r)
            UV = U_sel * V_sel  # (N, r)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check error
            err = _compute_error_vectorized(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-8:
                break
            
            # Decay learning rate
            if iteration % 100 == 0 and iteration > 0:
                lr *= 0.9
        
        if best_err < 1e-6:
            # Try to round to integers/half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error_vectorized(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-6:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Start from a reasonable lower bound. For 4x5x6, rank 120 is schoolbook.
    # Let's try from 80 upwards to save time, or maybe 60.
    # We will try a few specific ranks first that are likely candidates.
    candidate_ranks = [60, 70, 80, 90, 100, 110]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_decomposition(target_rank, num_attempts=10, max_iter=500, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    # If none found, try a linear scan from best_rank - 10 to best_rank
    if best_rank == n * m * p:
        for target_rank in range(110, 120):
            if time.time() > deadline - 10:
                break
            result = _try_rank_decomposition(target_rank, num_attempts=5, max_iter=200, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
