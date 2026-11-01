"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen + 1 schoolbook."""
    n, m, p = DIMENSIONS  # 2, 2, 15
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    # 7 blocks of size 2
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
    
    # 1 block of size 1
    start_k = 14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + start_k, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[start_k * n + i, col_offset:col_offset+4] = W_sb[i, :]
    col_offset += 4
    
    return U.tolist(), V.tolist(), W.tolist()

def _find_m225_rank14(timeout=60):
    """
    Attempt to find a rank-14 decomposition for M_{2,2,5}.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 2, 2, 5
    rank = 14
    num_params = (n*m + m*p + p*n) * rank  # 4*14 + 10*14 + 10*14 = 56 + 140 + 140 = 336
    
    def residuals(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
        W = params[(n*m+m*p)*rank:].reshape(p*n, rank)
        
        res = np.zeros(n*p)  # 2*5 = 10 output entries
        idx = 0
        for i in range(n):
            for k in range(p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = 0.0
                for j in range(m):
                    val += np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                res[idx] = val - 1.0
                idx += 1
        return res

    def objective(params):
        res = residuals(params)
        return np.sum(res**2)

    best_params = None
    best_obj = np.inf
    start_time = time.time()
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > timeout:
            break
        np.random.seed(seed)
        # Initialize with a structured guess
        x0 = np.zeros(num_params)
        # Add small random perturbations
        x0 += np.random.randn(num_params) * 0.3
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
            
        if best_obj < 1e-12:
            break
    
    if best_params is None or best_obj > 1e-10:
        return None
        
    U = best_params[:n*m*rank].reshape(n*m, rank)
    V = best_params[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
    W = best_params[(n*m+m*p)*rank:].reshape(p*n, rank)
    
    # Snap to half-integers and verify
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    res = residuals(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
    if np.max(np.abs(res)) < 1e-6:
        return U_snap, V_snap, W_snap
    
    # Try without snapping
    res = residuals(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
    if np.max(np.abs(res)) < 1e-8:
        # Round to nearest integer if close
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        res_int = residuals(np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()]))
        if np.max(np.abs(res_int)) < 1e-6:
            return U_int, V_int, W_int
        return U, V, W
    
    return None

def _build_rank49_if_m225_found(m225_decomp):
    """
    Build a rank-49 decomposition using rank-14 for 2x2x5 and 5 Strassen blocks for 2x2x10.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    U_m225, V_m225, W_m225 = m225_decomp
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    rank_m225 = U_m225.shape[1]  # 14
    rank_m2210 = 5 * 7  # 35
    total_rank = rank_m225 + rank_m2210  # 49
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # First 5 columns: use m225 decomposition
    U[:, :rank_m225] = U_m225
    for j in range(m):
        for k in range(5):
            V[j * p + k, :rank_m225] = V_m225[j * 5 + k, :]
    for k in range(5):
        for i in range(n):
            W[k * n + i, :rank_m225] = W_m225[k * 2 + i, :]
    
    # Next 10 columns (5 to 14): use 5 Strassen blocks
    col_offset = rank_m225
    for b in range(5):
        start_k = 5 + 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with safety margin
    
    # Try to find rank-14 decomposition for 2x2x5
    m225 = _find_m225_rank14(timeout=90)
    
    if m225 is not None and time.time() < deadline - 5:
        U, V, W = _build_rank49_if_m225_found(m225)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-53 solution
    U, V, W = _build_rank53()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

