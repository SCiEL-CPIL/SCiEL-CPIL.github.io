import numpy as np
import time
import warnings

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p
    U_school = np.zeros((dim_u, best_rank))
    V_school = np.zeros((dim_v, best_rank))
    W_school = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    max_rank_to_try = 60
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for R in range(32, max_rank_to_try + 1):
            if time.time() > deadline:
                break
            
            restarts = 10
            for r_idx in range(restarts):
                if time.time() > deadline:
                    break
                
                U = np.random.randn(dim_u, R)
                V = np.random.randn(dim_v, R)
                W = np.random.randn(dim_w, R)
                
                converged = False
                for it in range(100):
                    T_u = T.reshape(dim_u, dim_v * dim_w)
                    M = (V * W).reshape(R, dim_v * dim_w)
                    try:
                        U_new = np.linalg.lstsq(M.T, T_u.T, rcond=None)[0].T
                    except:
                        continue
                    
                    T_v = np.transpose(T, (1, 0, 2)).reshape(dim_v, dim_u * dim_w)
                    M_v = (U * W).reshape(R, dim_u * dim_w)
                    try:
                        V_new = np.linalg.lstsq(M_v.T, T_v.T, rcond=None)[0].T
                    except:
                        continue
                    
                    T_w = np.transpose(T, (2, 0, 1)).reshape(dim_w, dim_u * dim_v)
                    M_w = (U * V).reshape(R, dim_u * dim_v)
                    try:
                        W_new = np.linalg.lstsq(M_w.T, T_w.T, rcond=None)[0].T
                    except:
                        continue
                    
                    U, V, W = U_new, V_new, W_new
                    
                    T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                    err = np.linalg.norm(T - T_rec)
                    if err < 1e-8:
                        converged = True
                        break
                
                if converged:
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    T_rec_int = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
                    if np.allclose(T, T_rec_int, atol=1e-9):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }

    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
