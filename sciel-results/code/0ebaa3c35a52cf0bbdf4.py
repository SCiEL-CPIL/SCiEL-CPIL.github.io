import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    import numpy as np
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    # T[i*m+j, j*p+k, k*n+i]
    # We want sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1 if (i,j,k) is a valid entry, else 0?
    # No, the tensor T is 1 at (i,j,k) for all i,j,k.
    # The matrix multiplication tensor is T[i,j,k] = 1.
    # So we check if the sum is 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempts to reduce the rank by zeroing out columns and re-solving.
    """
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W

    # We want to keep `target_rank` columns.
    # Let's try to zero out the last (R0 - target_rank) columns.
    # This is a heuristic. We can try different subsets of columns to keep.
    # For now, we try keeping the first `target_rank` columns.
    
    U_keep = Ua[:, :target_rank].copy()
    V_keep = Va[:, :target_rank].copy()
    W_keep = Wa[:, :target_rank].copy()
    
    # We need to adjust U_keep, V_keep, W_keep such that the product matches the tensor.
    # This is a non-linear problem.
    # We can use gradient descent on the entries of U_keep, V_keep, W_keep.
    
    # Flatten the tensor target
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Define the loss function
    def loss(params):
        # params is a flattened array of size 3 * (n*m*p) * target_rank? No.
        # U_keep: (n*m, target_rank)
        # V_keep: (m*p, target_rank)
        # W_keep: (p*n, target_rank)
        # Total size: (n*m + m*p + p*n) * target_rank
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Compute tensor
        T = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
        # T shape: (n, m, p)
        # We want T to be all 1s.
        err = T - 1.0
        return np.sum(err**2)

    # Initialize params
    init_U = U_keep.flatten()
    init_V = V_keep.flatten()
    init_W = W_keep.flatten()
    init_params = np.concatenate([init_U, init_V, init_W])
    
    # Use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    res = minimize(loss, init_params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
    
    if time.time() > deadline - 5:
        return U, V, W
        
    if res.fun < 1e-8:
        # Extract new factors
        new_U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
        new_V = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        new_W = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Round to half-integers
        def round_half(a):
            return np.round(2 * a) / 2.0
            
        new_U = round_half(new_U)
        new_V = round_half(new_V)
        new_W = round_half(new_W)
        
        # Verify
        if _validate(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
            return new_U.tolist(), new_V.tolist(), new_W.tolist()
            
    # Try a second pass with tighter tolerance if we got close
    if res.fun < 1e-4:
        # Use the result as new initialization for a tighter optimization
        res2 = minimize(loss, res.x, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
        
        if time.time() > deadline - 5:
            return U, V, W
            
        if res2.fun < 1e-10:
            new_U = res2.x[:n*m*target_rank].reshape(n*m, target_rank)
            new_V = res2.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            new_W = res2.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            new_U = round_half(new_U)
            new_V = round_half(new_V)
            new_W = round_half(new_W)
            
            if _validate(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                return new_U.tolist(), new_V.tolist(), new_W.tolist()
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Start from known 66
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known

    # Try to reduce to 64
    if time.time() < deadline - 20:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 64, deadline)
        if U_new is not U: # Check if it changed
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Try to reduce to 62
    if time.time() < deadline - 20:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 62, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try to reduce to 60
    if time.time() < deadline - 20:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 60, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
