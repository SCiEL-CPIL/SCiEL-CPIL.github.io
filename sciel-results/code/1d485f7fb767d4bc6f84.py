import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor entries
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_set = set(target_indices)
    
    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school
    
    # Precompute vectorized target slices for ALS
    M_slices_a = [np.zeros(mp * np_) for _ in range(nm)]
    M_slices_b = [np.zeros(nm * np_) for _ in range(mp)]
    M_slices_c = [np.zeros(nm * mp) for _ in range(np_)]
    
    for a, b, c in target_indices:
        M_slices_a[a][b * np_ + c] = 1.0
        M_slices_b[b][a * np_ + c] = 1.0
        M_slices_c[c][a * mp + b] = 1.0

    def snap_to_half_int(matrix):
        snapped = []
        for row in matrix:
            new_row = []
            for val in row:
                v2 = val * 2
                r2 = round(v2)
                new_row.append(r2 / 2.0)
            snapped.append(new_row)
        return snapped

    def check_snapped_validity(U, V, W, R):
        Uf = np.array(U)
        Vf = np.array(V)
        Wf = np.array(W)
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        
        for a, b, c in target_indices:
            if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                return False
        
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if (a, b, c) in target_set:
                        continue
                    if abs(T_rec[a, b, c]) > 1e-9:
                        return False
        return True

    # Full residual function for least_squares
    def full_residual(params, R):
        U = params[:R*nm].reshape(R, nm)
        V = params[R*nm : R*nm + R*mp].reshape(R, mp)
        W = params[R*nm + R*mp : ].reshape(R, np_)
        
        # Compute T_rec
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        res = []
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if (a, b, c) in target_set:
                        res.append(T_rec[a, b, c] - 1.0)
                    else:
                        res.append(T_rec[a, b, c])
        return np.array(res)

    def polish_solution(U, V, W, R):
        # Flatten
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        num_params = len(params)
        num_residuals = nm * mp * np_
        
        try:
            if num_params <= num_residuals:
                method = 'lm'
            else:
                method = 'trf'
                
            result = least_squares(full_residual, params, args=(R,), method=method, max_nfev=10000)
            
            if result.cost < 1e-12:
                U_sol = result.x[:R*nm].reshape(R, nm)
                V_sol = result.x[R*nm : R*nm + R*mp].reshape(R, mp)
                W_sol = result.x[R*nm + R*mp : ].reshape(R, np_)
                
                # Snap
                U_snap = snap_to_half_int(U_sol.tolist())
                V_snap = snap_to_half_int(V_sol.tolist())
                W_snap = snap_to_half_int(W_sol.tolist())
                
                if check_snapped_validity(U_snap, V_snap, W_snap, R):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
        except Exception:
            pass
        return None

    # ALS solver
    def solve_rank(R, num_restarts=20):
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                return None
            
            # Random initialization
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(np_, R)
            
            # Scale
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(50): # More iterations
                if time.time() - start_time > timeout:
                    return None
                
                # Update U
                for a in range(nm):
                    K_a = np.zeros((mp * np_, R))
                    for r in range(R):
                        K_a[:, r] = np.kron(V[:, r], W[:, r])
                    u_a, _, _, _ = np.linalg.lstsq(K_a, M_slices_a[a], rcond=None)
                    U[a, :] = u_a
                
                # Update V
                for b in range(mp):
                    K_b = np.zeros((nm * np_, R))
                    for r in range(R):
                        K_b[:, r] = np.kron(U[:, r], W[:, r])
                    v_b, _, _, _ = np.linalg.lstsq(K_b, M_slices_b[b], rcond=None)
                    V[b, :] = v_b
                
                # Update W
                for c in range(np_):
                    K_c = np.zeros((nm * mp, R))
                    for r in range(R):
                        K_c[:, r] = np.kron(U[:, r], V[:, r])
                    w_c, _, _, _ = np.linalg.lstsq(K_c, M_slices_c[c], rcond=None)
                    W[c, :] = w_c
            
            # Check approximate validity
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(np_):
                        if (a, b, c) in target_set:
                            err = abs(T_rec[a, b, c] - 1.0)
                        else:
                            err = abs(T_rec[a, b, c])
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-3: # Candidate found
                polished = polish_solution(U, V, W, R)
                if polished:
                    return polished
                # If polish fails, try snapping ALS result directly
                U_snap = snap_to_half_int(U.tolist())
                V_snap = snap_to_half_int(V.tolist())
                W_snap = snap_to_half_int(W.tolist())
                if check_snapped_validity(U_snap, V_snap, W_snap, R):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # Try ranks from 1 to 30
    for R in range(1, R_school + 1):
        if time.time() - start_time > timeout:
            break
            
        # More restarts for lower ranks
        restarts = 50 if R < 30 else 10
        result = solve_rank(R, num_restarts=restarts)
        if result is not None:
            return result
            
    return best_result
