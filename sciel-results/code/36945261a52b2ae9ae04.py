import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-8):
    """Fast verification using vectorized numpy operations."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n*m, rank) or V_mat.shape != (m*p, rank) or W_mat.shape != (p*n, rank):
        return False

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _get_target_tensor(n, m, p):
    """Construct the all-ones tensor of shape (n, m, p)."""
    return np.ones((n, m, p))

def _objective(params, n, m, p, r):
    """
    Objective function for L-BFGS-B.
    params: flattened U and V.
    Returns: Frobenius norm squared of the residual tensor.
    """
    U_size = n * m * r
    V_size = m * p * r
    
    U_flat = params[:U_size]
    V_flat = params[U_size:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    
    total_residual = 0.0
    
    for i in range(n):
        for k in range(p):
            # A_ik[j, c] = U[i, j, c] * V[j, k, c]
            A_ik = U[i, :, :] * V[:, k, :]
            
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                diff = A_ik @ w - 1.0
                total_residual += np.dot(diff, diff)
            except:
                return 1e10
    
    return total_residual

def _optimize_rank(n, m, p, r, start_params, start_time, deadline):
    """
    Try to optimize U and V for a given rank r, starting from start_params.
    Returns (success, U, V, W) if success.
    """
    if time.time() > deadline - 10:
        return False, None, None, None
    
    res = minimize(_objective, start_params, args=(n, m, p, r), method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
    
    if time.time() > deadline - 10:
        return False, None, None, None
    
    if res.fun < 1e-6:
        U_size = n * m * r
        V_size = m * p * r
        U_flat = res.x[:U_size]
        V_flat = res.x[U_size:]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        
        # Reconstruct W
        W = np.zeros((p, n, r))
        for i in range(n):
            for k in range(p):
                A_ik = U[i, :, :] * V[:, k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    W[k, i, :] = w
                except:
                    pass
        
        # Convert to required format
        U_out = U.reshape(n*m, r).tolist()
        V_out = V.reshape(m*p, r).tolist()
        W_out = W.reshape(p*n, r).tolist()
        
        # Final verification
        if _verify_fast(U_out, V_out, W_out, n, m, p, trials=10, tol=1e-7):
            return True, U_out, V_out, W_out
    
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the standard Strassen decomposition (Rank 56)
    def _build_strassen_2x2x2():
        U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, 1]
        ], dtype=float)
        V = np.array([
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, -1, 0],
            [0, -1, 0, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
        return U, V, W

    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # Build the full 56-rank decomposition
    rank_56 = 56
    U_56 = np.zeros((n*m, rank_56))
    V_56 = np.zeros((m*p, rank_56))
    W_56 = np.zeros((p*n, rank_56))
    
    col_offset = 0
    # Left part: A_left (cols 0,1 of A) * B_top (rows 0,1 of B)
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U_56[r, col] = U_s[f, r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = V_s[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = W_s[r, f]
        col_offset += 7
        
    # Right part: A_right (cols 2,3 of A) * B_bot (rows 2,3 of B)
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U_56[2 + r, col] = U_s[f, r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = V_s[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = W_s[r, f]
        col_offset += 7
    
    # Verify Rank 56
    if not _verify_fast(U_56.tolist(), V_56.tolist(), W_56.tolist(), n, m, p, trials=5, tol=1e-9):
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n*m, schoolbook_rank))
        V_sb = np.zeros((m*p, schoolbook_rank))
        W_sb = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U = U_56.tolist()
    best_V = V_56.tolist()
    best_W = W_56.tolist()
    best_rank = 56
    
    # 2. Try to reduce rank using optimization
    for r in range(55, 29, -1):
        if time.time() > deadline - 15:
            break
            
        U_init = U_56[:, :r].flatten()
        V_init = V_56[:, :r].flatten()
        params_init = np.concatenate([U_init, V_init])
        
        success, U_opt, V_opt, W_opt = _optimize_rank(n, m, p, r, params_init, start_time, deadline)
        
        if success:
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            best_rank = r
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_decomposition(n, m, p):
    """Build a block-Strassen decomposition."""
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    col_offset = 0
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[r][col] = U_s[f][r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[2 + r][col] = U_s[f][r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_linear_reduction(U, V, W, n, m, p, start_time, deadline):
    """Placeholder for the old reduction method."""
    return U, V, W
