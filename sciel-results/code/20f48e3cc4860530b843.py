import numpy as np
import time
from scipy.optimize import least_squares, minimize

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 135.0
    
    # Try to find rank 50 decomposition (5 blocks of 2x2x3 rank 10)
    result = _try_rank50(deadline)
    if result is not None:
        return result
        
    # Fallback to rank 53
    return _build_rank53()

def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7
        
    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]
            
    return {"U": U, "V": V, "W": W}

def _try_rank50(deadline):
    n, m, p = 2, 2, 15
    rank_block = 10
    num_blocks = 5
    total_rank = rank_block * num_blocks
    
    # Find a rank 10 decomposition for 2x2x3
    U_b, V_b, W_b = _find_m223_rank10(deadline)
    if U_b is None:
        return None
        
    # Assemble 5 blocks
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        k0 = 3 * b
        col_offset = b * rank_block
        for r in range(rank_block):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_b[i*2 + j][r]
            for j in range(2):
                for k in range(3):
                    kk = k0 + k
                    V[j*p + kk][c] = V_b[j*3 + k][r]
            for k in range(3):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_b[k*2 + i][r]
                    
    return {"U": U, "V": V, "W": W}

def _find_m223_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10
    
    # Residual function for 2x2x3
    def residuals(params):
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+6*rank].reshape(6, rank)
        W = params[4*rank+6*rank:].reshape(6, rank)
        
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
    
    # Try multiple seeds with diverse initialization strategies
    for seed in range(80):
        if time.time() > deadline - 25:
            break
            
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed % 3 == 0:
            # Small random
            x0 = np.random.randn(16*rank) * 0.3
        elif seed % 3 == 1:
            # Medium random
            x0 = np.random.randn(16*rank) * 0.8
        else:
            # Structured: start from near-zero with some pattern
            x0 = np.random.randn(16*rank) * 0.5
            # Add some structure to help convergence
            x0[:4*rank] += 0.2 * np.eye(4, rank)[:, :4].flatten()
            x0[4*rank:4*rank+6*rank] += 0.2 * np.eye(6, rank)[:, :6].flatten()
            x0[4*rank+6*rank:] += 0.2 * np.eye(6, rank)[:, :6].flatten()
        
        try:
            # First try LM (Levenberg-Marquardt)
            result = least_squares(residuals, x0, method='lm', max_nfev=3000)
            obj_val = np.sum(result.fun**2)
            
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = result.x.copy()
            
            if obj_val < 1e-10:
                break
                
        except Exception:
            pass
        
        # If LM didn't converge well, try L-BFGS-B as fallback
        if time.time() > deadline - 25:
            break
        if best_obj > 1e-8:
            try:
                result2 = minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'gtol': 1e-10})
                obj_val2 = result2.fun
                if obj_val2 < best_obj:
                    best_obj = obj_val2
                    best_params = result2.x.copy()
                if obj_val2 < 1e-10:
                    break
            except Exception:
                pass
    
    if best_params is None or best_obj > 1e-8:
        return None, None, None
    
    U = best_params[:4*rank].reshape(4, rank)
    V = best_params[4*rank:4*rank+6*rank].reshape(6, rank)
    W = best_params[4*rank+6*rank:].reshape(6, rank)
    
    # Final polish pass: high-precision refinement
    if time.time() < deadline - 20:
        try:
            polished = least_squares(residuals, best_params, method='lm', 
                                     max_nfev=5000, xtol=1e-15, ftol=1e-15)
            if np.sum(polished.fun**2) < best_obj:
                best_params = polished.x
        except Exception:
            pass
    
    U = best_params[:4*rank].reshape(4, rank)
    V = best_params[4*rank:4*rank+6*rank].reshape(6, rank)
    W = best_params[4*rank+6*rank:].reshape(6, rank)
    
    # Snap to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify
    res_r = residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
    if np.max(np.abs(res_r)) < 1e-6:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try snapping to integers as well
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    res_i = residuals(np.concatenate([U_i.flatten(), V_i.flatten(), W_i.flatten()]))
    if np.max(np.abs(res_i)) < 1e-6:
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try quarter-integer snapping
    U_q = np.round(U * 4) / 4
    V_q = np.round(V * 4) / 4
    W_q = np.round(W * 4) / 4
    res_q = residuals(np.concatenate([U_q.flatten(), V_q.flatten(), W_q.flatten()]))
    if np.max(np.abs(res_q)) < 1e-6:
        return U_q.tolist(), V_q.tolist(), W_q.tolist()
    
    return None, None, None
