"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2.
    Returns U (4x7), V (4x7), W (4x7) with W indexed by [k*2+i, r]."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
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

    # Last column: schoolbook for k=14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def _find_m225_rank15(deadline):
    """
    Find a rank-15 decomposition for M_{2,2,5}.
    Uses multiple restarts with structured and random initializations.
    """
    n, m, p = 2, 2, 5
    rank = 15
    
    def residuals(params):
        U = params[:60].reshape(4, 15)
        V = params[60:120].reshape(10, 15)
        W = params[120:180].reshape(10, 15)
        
        res = np.zeros(20)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    res[idx] = np.dot(U[i*2+j], V[j*5+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
        
    best_params = None
    best_obj = np.inf
    
    # Structured initialization: 2 Strassen blocks (rank 14) + 1 schoolbook (rank 4) = rank 18, then project to rank 15
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    # Build a rank-18 solution first
    U18 = np.zeros((4, 18))
    V18 = np.zeros((10, 18))
    W18 = np.zeros((10, 18))
    
    # Two Strassen blocks for k=0,1 and k=2,3
    for b in range(2):
        start_k = 2 * b
        U18[:, b*7:(b+1)*7] = U_str
        for j in range(2):
            for k_local in range(2):
                k_global = start_k + k_local
                V18[j*5+k_global, b*7:(b+1)*7] = V_str[j*2+k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(2):
                W18[k_global*2+i, b*7:(b+1)*7] = W_str[k_local*2+i, :]
    
    # Schoolbook for k=4
    U18[:, 14:18] = U_sb
    for j in range(2):
        V18[j*5+4, 14:18] = V_sb[j, :]
    for i in range(2):
        W18[4*2+i, 14:18] = W_sb[i, :]
    
    # Try to reduce to rank 15 by zeroing out some columns and re-optimizing
    # Strategy: start with rank-18, zero out 3 columns, then optimize
    x0_structured = np.concatenate([U18[:, :15].flatten(), V18[:, :15].flatten(), W18[:, :15].flatten()])
    
    seeds = [x0_structured]
    for seed in range(30):
        np.random.seed(seed)
        # Random initialization with small values
        seeds.append(np.random.randn(180) * 0.3)
    
    for x0 in seeds:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is None or best_obj > 1e-8:
        return None, None, None
        
    U = best_params[:60].reshape(4, 15)
    V = best_params[60:120].reshape(10, 15)
    W = best_params[120:180].reshape(10, 15)
    
    # Polish pass: tighter optimization from the best solution
    if time.time() < deadline - 2:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-16})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x.copy()
                U = best_params[:60].reshape(4, 15)
                V = best_params[60:120].reshape(10, 15)
                W = best_params[120:180].reshape(10, 15)
        except Exception:
            pass
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    res_snap = residuals(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap
        
    # If snapping fails, check if raw is close to half-integer
    if np.allclose(U*2, np.round(U*2), atol=1e-6) and \
       np.allclose(V*2, np.round(V*2), atol=1e-6) and \
       np.allclose(W*2, np.round(W*2), atol=1e-6):
        return U_snap, V_snap, W_snap

    # Last resort: if the raw solution is very accurate, try integer rounding
    if np.max(np.abs(residuals(best_params))) < 1e-10:
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        res_int = residuals(np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()]))
        if np.max(np.abs(res_int)) < 1e-8:
            return U_int, V_int, W_int

    return None, None, None

def _try_rank50(deadline):
    """
    Try to find a rank-50 decomposition:
    5 Strassen blocks (rank 35) for k=0..9 + M_{2,2,5} rank-15 for k=10..14
    """
    n, m, p = DIMENSIONS
    
    # Find rank-15 M_{2,2,5}
    U225, V225, W225 = _find_m225_rank15(deadline - 20)
    
    if U225 is None:
        return None, None, None
    
    # Assemble the full rank-50 decomposition
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    # Fill first 5 blocks with Strassen (k=0..9)
    col_offset = 0
    for b in range(5):
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
        
    # Fill last block with M_{2,2,5} (k=10..14)
    U[:, col_offset:col_offset+15] = U225
    for j in range(m):
        for k_local in range(5):
            k_global = 10 + k_local
            V[j * p + k_global, col_offset:col_offset+15] = V225[j * 5 + k_local, :]
    for k_local in range(5):
        k_global = 10 + k_local
        for i in range(n):
            W[k_global * n + i, col_offset:col_offset+15] = W225[k_local * 2 + i, :]
            
    return U, V, W

def _find_m223_rank10(deadline):
    """
    Find a rank-10 decomposition for M_{2,2,3}.
    Uses a global optimization strategy with multiple random restarts.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    def residuals(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
        
    best_params = None
    best_obj = np.inf
    
    # Use a mix of structured and random initializations
    # Structured: Start from rank-11 (Strassen 7 + Schoolbook 4) and try to merge
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    U11[:, 0:7] = U_str
    for j in range(2):
        for k_local in range(2):
            V11[j*3+k_local, 0:7] = V_str[j*2+k_local, :]
    for k_local in range(2):
        for i in range(2):
            W11[k_local*2+i, 0:7] = W_str[k_local*2+i, :]
    U11[:, 7:11] = U_sb
    for j in range(2):
        V11[j*3+2, 7:11] = V_sb[j, :]
    for i in range(2):
        W11[2*2+i, 7:11] = W_sb[i, :]
        
    x0_structured = np.concatenate([U11.flatten(), V11.flatten(), W11.flatten()])
    
    seeds = [x0_structured]
    for seed in range(20):
        np.random.seed(seed)
        seeds.append(np.random.randn(160) * 0.5)
        
    for x0 in seeds:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-10:
                break
        except Exception:
            continue
            
    if best_params is None or best_obj > 1e-8:
        return None, None, None
        
    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    res_snap = residuals(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap
        
    # If snapping fails, check if raw is close to half-integer
    if np.allclose(U*2, np.round(U*2), atol=1e-6) and \
       np.allclose(V*2, np.round(V*2), atol=1e-6) and \
       np.allclose(W*2, np.round(W*2), atol=1e-6):
        return U_snap, V_snap, W_snap

    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U53, V53, W53 = _build_schoolbook(n, m, p)
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-50 solution (5 Strassen + rank-15 M_{2,2,5})
    if time.time() < deadline - 5:
        U50, V50, W50 = _try_rank50(deadline)
        if U50 is not None:
            if _verify_decomposition(U50, V50, W50, n, m, p, tol=1e-6):
                return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
# EVOLVE-BLOCK-END
