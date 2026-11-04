import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 145.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    r = r2 * r2  # 49
    
    U2_3d = U2.reshape(2, 2, r2)
    V2_3d = V2.reshape(2, 2, r2)
    W2_3d = W2.reshape(2, 2, r2)
    
    U4_5d = np.einsum('abk,cdl->acbdkl', U2_3d, U2_3d)
    U4 = U4_5d.reshape(16, 49).astype(np.int16)
    V4 = np.einsum('abk,cdl->acbdkl', V2_3d, V2_3d).reshape(16, 49).astype(np.int16)
    W4 = np.einsum('abk,cdl->acbdkl', W2_3d, W2_3d).reshape(16, 49).astype(np.int16)
    
    return U4, V4, W4

def _build_4x4x8():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98."""
    U4, V4, W4 = _build_4x4x4_strassen()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8, seed=42, tol=1e-9):
    """Verify the tensor decomposition on random test matrices."""
    np.random.seed(seed)
    A = np.random.randint(-4, 5, size=(n, m)).astype(np.float64)
    B = np.random.randint(-4, 5, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    rank = U.shape[1]
    C_comp = np.zeros((n, p), dtype=np.float64)
    for r in range(rank):
        Ar = A * U[:, r].reshape(n, m)
        Br = B * V[:, r].reshape(m, p)
        Mr = Ar @ Br
        Wr = W[:, r].reshape(p, n).T
        C_comp += Mr * Wr
    err = np.max(np.abs(C_comp - C_expected))
    return err < tol, err

def _optimize_rank(U, V, W, n=4, m=4, p=8, time_budget=5.0):
    """
    Try to reduce rank by 1 by optimizing and zeroing one column.
    Returns (U, V, W, success)
    """
    start_time = time.time()
    rank = U.shape[1]
    
    # Convert to float
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
    success = False
    
    # Try removing each column
    for c in range(rank):
        if time.time() - start_time > time_budget:
            break
            
        # Create mask to exclude column c
        mask = np.ones(rank, dtype=bool)
        mask[c] = False
        U_sub = U_f[:, mask]
        V_sub = V_f[:, mask]
        W_sub = W_f[:, mask]
        
        # Parameters to optimize
        # We fix the structure? No, we optimize all remaining entries.
        # This is a large space. 
        # To make it feasible, we might only optimize a subset or use a coarse grid?
        # Given the time, we'll try a quick L-BFGS-B step.
        
        x0 = np.concatenate([U_sub.flatten(), V_sub.flatten(), W_sub.flatten()])
        
        def objective(params):
            U_opt = params[:16*(rank-1)].reshape(16, rank-1)
            V_opt = params[16*(rank-1):16*(rank-1)+32*(rank-1)].reshape(32, rank-1)
            W_opt = params[16*(rank-1)+32*(rank-1):].reshape(32, rank-1)
            
            # Compute residual on a small set of test matrices
            # Use fixed seeds for speed
            np.random.seed(1)
            A = np.random.randint(-2, 3, size=(4, 4)).astype(np.float64)
            B = np.random.randint(-2, 3, size=(4, 8)).astype(np.float64)
            C_exp = A @ B
            
            C_comp = np.zeros((4, 8), dtype=np.float64)
            for r in range(rank-1):
                Ar = A * U_opt[:, r].reshape(4, 4)
                Br = B * V_opt[:, r].reshape(4, 8)
                Mr = Ar @ Br
                Wr = W_opt[:, r].reshape(8, 4).T
                C_comp += Mr * Wr
                
            return np.sum((C_comp - C_exp)**2)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 5, 'maxfun': 10})
            if result.fun < 1e-6:
                # Check if we can round to half-integers
                U_opt = result.x[:16*(rank-1)].reshape(16, rank-1)
                V_opt = result.x[16*(rank-1):16*(rank-1)+32*(rank-1)].reshape(32, rank-1)
                W_opt = result.x[16*(rank-1)+32*(rank-1):].reshape(32, rank-1)
                
                # Round to nearest 0.5
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                
                # Verify exactness with higher precision
                ok, err = _verify(U_r.astype(np.int16) if np.allclose(U_r, np.round(U_r)) else U_r,
                                  V_r.astype(np.int16) if np.allclose(V_r, np.round(V_r)) else V_r,
                                  W_r.astype(np.int16) if np.allclose(W_r, np.round(W_r)) else W_r)
                
                if ok:
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    success = True
                    break # Found a valid reduction
        except Exception:
            pass
            
    return best_U, best_V, best_W, success

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    U, V, W = _build_4x4x8()
    
    ok, err = _verify(U, V, W)
    if not ok:
        # Fallback to trivial
        n, m, p = 4, 4, 8
        rank = n * m * p
        U_f = np.zeros((n * m, rank), dtype=np.int16)
        V_f = np.zeros((m * p, rank), dtype=np.int16)
        W_f = np.zeros((p * n, rank), dtype=np.int16)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    current_U, current_V, current_W = U, V, W
    current_rank = U.shape[1]
    
    # Iterative reduction
    while _time_left() > 10:
        U_opt, V_opt, W_opt, success = _optimize_rank(current_U, current_V, current_W, time_budget=5.0)
        if success:
            current_U, current_V, current_W = U_opt, V_opt, W_opt
            current_rank -= 1
        else:
            break
            
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
