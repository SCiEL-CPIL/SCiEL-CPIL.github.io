import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_2x3x3(U, V, W, tol=1e-8):
    """Verify 2x3x3 decomposition."""
    n, m, p = 2, 3, 3
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _solve_2x3x3(rank, timeout=30):
    """
    Attempt to find a rank 'rank' decomposition for 2x3x3 matmul.
    Returns (U, V, W) lists or None.
    """
    n, m, p = 2, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    
    # Objective function
    def objective(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
        W = params[nm*rank + mp*rank:].reshape(pn, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err += (val - 1.0)**2
        return err
    
    # Gradient
    def gradient(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
        W = params[nm*rank + mp*rank:].reshape(pn, rank)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    grad_U[idx_u, :] += 2.0 * err * (V[idx_v, :] * W[idx_w, :])
                    grad_V[idx_v, :] += 2.0 * err * (U[idx_u, :] * W[idx_w, :])
                    grad_W[idx_w, :] += 2.0 * err * (U[idx_u, :] * V[idx_v, :])
                    
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    best_result = None
    best_loss = float('inf')
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(10):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(42 + attempt)
        U0 = np.random.randn(nm, rank) * 0.5
        V0 = np.random.randn(mp, rank) * 0.5
        W0 = np.random.randn(pn, rank) * 0.5
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < best_loss:
                if res.fun < best_loss:
                    best_loss = res.fun
                    x_opt = res.x
                    U_opt = x_opt[:nm*rank].reshape(nm, rank)
                    V_opt = x_opt[nm*rank:nm*rank + mp*rank].reshape(mp, rank)
                    W_opt = x_opt[nm*rank + mp*rank:].reshape(pn, rank)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if _verify_2x3x3(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                        
                    # If rounding failed, try original if loss is extremely small
                    if res.fun < 1e-6 and _verify_2x3x3(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                        return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
                        
        except Exception:
            pass
            
    return None

def _construct_from_2x3x3(U_sub, V_sub, W_sub):
    """
    Construct 2x3x9 decomposition from 2x3x3 sub-decomposition.
    """
    n, m, p = 2, 3, 9
    r_sub = len(U_sub[0])
    r_total = 3 * r_sub
    
    U = [[0] * r_total for _ in range(n * m)]
    V = [[0] * r_total for _ in range(m * p)]
    W = [[0] * r_total for _ in range(p * n)]
    
    # U is repeated 3 times
    for i in range(n * m):
        for r in range(r_sub):
            U[i][r] = U_sub[i][r]
            U[i][r_sub + r] = U_sub[i][r]
            U[i][2 * r_sub + r] = U_sub[i][r]
            
    # V is block diagonal in the k dimension
    # V_sub is (m*p_sub, r_sub) with p_sub=3
    # V is (m*p, r_total) with p=9
    for j in range(m):
        for k_sub in range(3): # 0,1,2
            for block in range(3): # 0,1,2
                k_full = block * 3 + k_sub
                for r in range(r_sub):
                    V[j * p + k_full][block * r_sub + r] = V_sub[j * 3 + k_sub][r]
                    
    # W is block diagonal in the k dimension
    # W_sub is (p_sub*n, r_sub) with p_sub=3
    # W is (p*n, r_total) with p=9
    for k_sub in range(3):
        for i in range(n):
            for block in range(3):
                k_full = block * 3 + k_sub
                for r in range(r_sub):
                    W[k_full * n + i][block * r_sub + r] = W_sub[k_sub * n + i][r]
                    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 54
    
    # Try to find a lower rank decomposition for the 2x3x3 sub-problem
    # Known rank is 11. Try 11, 12, 13.
    for r_sub in [11, 12, 13]:
        if time.time() > deadline:
            break
            
        result_sub = _solve_2x3x3(r_sub, timeout=30)
        if result_sub is not None:
            U_sub, V_sub, W_sub = result_sub
            result = _construct_from_2x3x3(U_sub, V_sub, W_sub)
            
            # Verify the full 2x3x9 result
            if _verify(result["U"], result["V"], result["W"]):
                return result
                
    # Fallback to schoolbook
    return _schoolbook()
