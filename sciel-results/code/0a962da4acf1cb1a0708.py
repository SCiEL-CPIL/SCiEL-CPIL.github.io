import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

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

def _strassen_44():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for (li, lj), val in u_coeffs.items():
                U[(li + xi) * m + (lj + xj)][col] += val
            for (lj, lk), val in v_coeffs.items():
                V[(lj + yj) * p + (lk + yk)][col] += val
            for (li, lk), val in w_coeffs.items():
                W[(lk + zk) * n + (li + zi)][col] += val
            col += 1
            
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_solution, n, m, p, target_rank):
    """
    Attempts to reduce the rank of the base solution to target_rank
    by zeroing out columns and using least squares to adjust the rest.
    """
    U_base = np.array(base_solution["U"], dtype=float)
    V_base = np.array(base_solution["V"], dtype=float)
    W_base = np.array(base_solution["W"], dtype=float)
    
    rank = U_base.shape[1]
    
    # Reshape for easier manipulation
    U_3d = U_base.reshape(n, m, rank)
    V_3d = V_base.reshape(m, p, rank)
    W_3d = W_base.reshape(p, n, rank)
    
    # Target tensor
    T_target = np.ones((n, m, p))
    
    # Try removing one column at a time
    for col_to_remove in range(rank):
        if target_rank >= rank:
            continue
            
        # Create masks for active columns
        active_cols = [c for c in range(rank) if c != col_to_remove]
        
        # Extract active parts
        U_act = U_3d[:, :, active_cols]
        V_act = V_3d[:, :, active_cols]
        W_act = W_3d[:, :, active_cols]
        
        # Flatten variables
        # U: n*m*target_rank, V: m*p*target_rank, W: p*n*target_rank
        U_flat = U_act.reshape(-1)
        V_flat = V_act.reshape(-1)
        W_flat = W_act.reshape(-1)
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def residuals(x):
            U_r = x[:len(U_flat)].reshape(n, m, target_rank)
            V_r = x[len(U_flat):len(U_flat)+len(V_flat)].reshape(m, p, target_rank)
            W_r = x[len(U_flat)+len(V_flat):].reshape(p, n, target_rank)
            
            T_calc = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return (T_calc - T_target).flatten()
        
        try:
            # Use least squares to minimize residuals
            # x_scale helps with numerical stability
            res = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if res.success and np.max(np.abs(res.fun)) < 1e-8:
                # Snap to half-integers
                # Check if 2*x is close to an integer
                U_sol_raw = res.x[:len(U_flat)].reshape(n, m, target_rank)
                V_sol_raw = res.x[len(U_flat):len(U_flat)+len(V_flat)].reshape(m, p, target_rank)
                W_sol_raw = res.x[len(U_flat)+len(V_flat):].reshape(p, n, target_rank)
                
                U_sol = 2 * np.rint(U_sol_raw * 2) / 2
                V_sol = 2 * np.rint(V_sol_raw * 2) / 2
                W_sol = 2 * np.rint(W_sol_raw * 2) / 2
                
                # Verify exactness
                T_final = np.einsum('ijc,jkc,kic->ijk', U_sol, V_sol, W_sol)
                if np.max(np.abs(T_final - T_target)) < 1e-9:
                    # Reconstruct full matrices with the removed column as 0
                    U_new = np.zeros((n, m, rank))
                    V_new = np.zeros((m, p, rank))
                    W_new = np.zeros((p, n, rank))
                    
                    for idx, c in enumerate(active_cols):
                        U_new[:, :, c] = U_sol[:, :, idx]
                        V_new[:, :, c] = V_sol[:, :, idx]
                        W_new[:, :, c] = W_sol[:, :, idx]
                        
                    return {
                        "U": U_new.reshape(n*m, rank).tolist(),
                        "V": V_new.reshape(m*p, rank).tolist(),
                        "W": W_new.reshape(p*n, rank).tolist()
                    }
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if not ok:
        return _schoolbook()
    
    # Try to reduce to 43
    if time.time() < deadline - 20:
        result_43 = _try_reduce_rank(result_44, n, m, p, 43)
        if result_43:
            ok, err = _verify_fast(result_43["U"], result_43["V"], result_43["W"], n, m, p)
            if ok:
                return result_43
                
    return result_44
