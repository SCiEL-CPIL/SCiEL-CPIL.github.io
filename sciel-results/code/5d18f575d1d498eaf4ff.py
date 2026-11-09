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

def _try_rank_100(deadline):
    """Attempt to find the known rank-100 decomposition for 4x5x6."""
    n, m, p = DIMENSIONS
    target_rank = 100
    
    # Precompute index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()  # (n*m*p,)
    idx_v = (jj * p + kk).flatten()  # (n*m*p,)
    idx_w = (kk * n + ii).flatten()  # (n*m*p,)
    N = n * m * p  # 120
    
    # We will use Adam optimizer for faster convergence
    # And a specific initialization strategy
    
    for attempt in range(50): # More attempts
        if time.time() > deadline - 20:
            return None
        
        np.random.seed(12345 + attempt)
        
        # Initialization: Start with random values, scaled appropriately
        # The solution is known to have small integer entries (mostly 0, 1, -1, 2)
        # So we initialize in that range
        
        # Strategy 1: Random integers in {-1, 0, 1} with some 2s?
        # Or better: Start from a "sparse" random structure and densify?
        
        # Let's try a "Gaussian" init first, then project?
        # Actually, for rank 100, the matrix is dense.
        
        # Use a structured init:
        # For each column c, pick a random (i, j, k) and set U,V,W to 1.
        # Then add small random noise to break symmetry.
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        for c in range(target_rank):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
            
        # Add noise to allow movement out of the discrete manifold
        U += np.random.randn(n*m, target_rank) * 0.1
        V += np.random.randn(m*p, target_rank) * 0.1
        W += np.random.randn(p*n, target_rank) * 0.1
        
        # Adam Optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(2000):
            if time.time() > deadline - 10:
                break
                
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            
            prod = U_sel * V_sel * W_sel
            vals = prod.sum(axis=1)
            residuals = vals - 1.0
            
            # Gradients
            VW = V_sel * W_sel
            UW = U_sel * W_sel
            UV = U_sel * V_sel
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            # Adam Update
            t = iteration + 1
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check error
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            vals = (U_sel * V_sel * W_sel).sum(axis=1)
            err = float(np.max(np.abs(vals - 1.0)))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
            # Decay LR
            if iteration % 500 == 0 and iteration > 0:
                lr *= 0.5
                
        # Polish: Round to nearest integers
        if best_err < 1e-4:
            U_r = np.round(best_U)
            V_r = np.round(best_V)
            W_r = np.round(best_W)
            
            err_r = _compute_error_vectorized(U_r, V_r, W_r)
            if err_r < 1e-10:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                
            # Try half-integers?
            U_h = np.round(best_U * 2) / 2
            V_h = np.round(best_V * 2) / 2
            W_h = np.round(best_W * 2) / 2
            err_h = _compute_error_vectorized(U_h, V_h, W_h)
            if err_h < 1e-10:
                return {
                    "U": U_h.tolist(),
                    "V": V_h.tolist(),
                    "W": W_h.tolist()
                }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    
    # Try to find the rank 100 decomposition
    result = _try_rank_100(deadline)
    if result is not None:
        best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
