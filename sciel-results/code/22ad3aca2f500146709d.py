import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 8]

def _get_target_tensor():
    """
    Constructs the target multiplication tensor M for 3x3 x 3x8.
    Shape: (9, 24, 24)
    Index u = i*3 + j
    Index v = j*8 + k
    Index w = k*3 + i
    M[u, v, w] = 1 if j(u)==j(v) and k(v)==k(w) and i(w)==i(u), else 0.
    """
    n, m, p = 3, 3, 8
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                M[u, v, w] = 1.0
    return M

def _laderman_3x3x3():
    """
    Returns U, V, W for Laderman's algorithm for 3x3x3 matrix multiplication.
    Rank 23.
    U: 9x23, V: 9x23, W: 9x23
    """
    terms = []
    terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]))
    terms.append(([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    terms.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]))
    terms.append(([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    terms.append(([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    terms.append(([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]))
    terms.append(([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    terms.append(([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0]))
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    terms.append(([1,0,0,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0], [-1,0,0,0,0,0,0,0,0]))
    terms.append(([0,1,0,0,0,1,0,0,0], [0,0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,0,1]))
    terms.append(([0,0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]))
    terms.append(([0,0,0,1,0,0,0,1,0], [0,1,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0,0]))
    terms.append(([1,1,0,0,1,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    terms.append(([0,0,0,0,1,1,0,0,1], [0,0,0,0,1,0,0,1,1], [0,0,0,0,0,0,0,0,0]))
    terms.append(([1,0,1,0,0,0,0,0,1], [1,0,0,0,0,0,1,0,1], [1,0,0,0,0,0,0,0,0]))
    terms.append(([0,0,0,1,1,0,0,1,0], [0,1,0,0,1,1,0,0,0], [0,0,0,1,0,0,0,0,0]))
    terms.append(([0,1,0,0,0,1,0,0,1], [0,0,0,1,0,0,0,1,1], [0,0,0,0,0,0,0,1,0]))
    terms.append(([0,0,1,1,0,0,0,1,0], [0,1,0,0,0,1,1,0,0], [0,0,0,0,1,0,0,0,0]))
    terms.append(([1,1,1,0,1,1,0,0,1], [1,0,0,1,0,1,1,1,1], [1,0,0,0,0,0,0,0,0]))
    terms.append(([1,1,0,1,1,0,1,1,0], [1,1,0,1,1,0,1,1,0], [0,0,0,0,0,0,0,0,1]))
    terms.append(([0,1,1,0,1,1,0,1,1], [0,0,0,1,0,1,1,1,1], [0,0,0,0,0,0,0,0,1]))
    terms.append(([0,1,1,1,0,1,1,1,0], [1,1,1,1,0,1,1,0,0], [0,0,0,0,0,0,0,0,1]))
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    for r in range(23):
        ca, cb, cc = terms[r]
        U[:, r] = ca
        V[:, r] = cb
        W[:, r] = cc
    
    return U, V, W

def _build_rank_64():
    n, m, p = DIMENSIONS
    U_lad, V_lad, W_lad = _laderman_3x3x3()
    
    rank_total = 23 + 23 + 18
    U_full = np.zeros((9, rank_total))
    V_full = np.zeros((24, rank_total))
    W_full = np.zeros((24, rank_total))
    
    # Block 1: Laderman for cols 0-2
    for r in range(23):
        U_full[:, r] = U_lad[:, r]
        for j in range(3):
            for k in range(3):
                V_full[j*8+k, r] = V_lad[j*3+k, r]
        W_full[:, r] = W_lad[:, r]
        
    # Block 2: Laderman for cols 3-5
    offset = 23
    for r in range(23):
        col = r + offset
        U_full[:, col] = U_lad[:, r]
        for j in range(3):
            for k in range(3):
                V_full[j*8+(k+3), col] = V_lad[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*3+i, col] = W_lad[k*3+i, r]
                
    # Block 3: Schoolbook for cols 6-7
    offset2 = 46
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_full[i*3+j, offset2+idx] = 1
                V_full[j*8+(k+6), offset2+idx] = 1
                W_full[(k+6)*3+i, offset2+idx] = 1
                idx += 1
                
    return U_full, V_full, W_full

def _try_reduce_rank(U, V, W, target_tensor, target_rank, time_limit=60):
    """
    Attempts to reduce the rank by optimizing coefficients of a subset of terms.
    """
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = 9, 24, 24
    current_rank = U.shape[1]
    
    # We will try to keep the first `target_rank` terms and optimize them.
    # This is a heuristic; dropping arbitrary terms might fail.
    # Ideally we should drop the "least useful" term.
    # For now, we drop the last term (from schoolbook).
    
    U_init = U[:, :target_rank].copy()
    V_init = V[:, :target_rank].copy()
    W_init = W[:, :target_rank].copy()
    
    # Flatten
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(x):
        U_curr = x[:dim_u*target_rank].reshape(dim_u, target_rank)
        V_curr = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
        W_curr = x[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
        
        # Compute tensor approximation
        # T_approx[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # Efficiently: T_approx = U @ (V * W.T).T ? No.
        # T_approx = np.einsum('ur,vr,wr->uvw', U_curr, V_curr, W_curr)
        T_approx = np.einsum('ur,vr,wr->uvw', U_curr, V_curr, W_curr)
        
        diff = T_approx - target_tensor
        return np.sum(diff**2)
    
    start_time = time.time()
    try:
        # Use L-BFGS-B for large sparse-ish problems? Or just default.
        # Since we have ~3500 variables, this might be slow.
        # We use a limited number of iterations.
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
        
        if res.success and res.fun < 1e-8:
            U_opt = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V_opt = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W_opt = res.x[dim_u*target_rank + dim_v*target_rank :].reshape(dim_w, target_rank)
            return U_opt, V_opt, W_opt
    except Exception as e:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    U64, V64, W64 = _build_rank_64()
    
    # Verify rank 64
    target_tensor = _get_target_tensor()
    T_approx = np.einsum('ur,vr,wr->uvw', U64, V64, W64)
    if not np.allclose(T_approx, target_tensor, atol=1e-9):
        # Fallback to schoolbook if construction failed
        return _schoolbook_full()
    
    # Try to reduce to rank 63
    if time.time() < deadline - 20.0:
        res_63 = _try_reduce_rank(U64, V64, W64, target_tensor, 63, time_limit=120)
        if res_63 is not None:
            U_opt, V_opt, W_opt = res_63
            # Final verification
            T_check = np.einsum('ur,vr,wr->uvw', U_opt, V_opt, W_opt)
            if np.allclose(T_check, target_tensor, atol=1e-8):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                
    return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

def _schoolbook_full():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
