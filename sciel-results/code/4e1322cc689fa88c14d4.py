def _try_rank23(start_time, deadline, idx_u, idx_v, idx_w, n, m, p):
    """Attempt to find a rank-23 decomposition via drop-and-fit from rank-26."""
    if not HAVE_SCIPY:
        return None
        
    R = 23
    R_start = 26
    
    # Construct rank-26 solution locally
    U26 = np.zeros((n * m, R_start))
    V26 = np.zeros((m * p, R_start))
    W26 = np.zeros((p * n, R_start))
    
    # Strassen 7 terms
    # m1
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    # m2
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    # m3
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    # m4
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    # m5
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    # m6
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    # m7
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Border terms
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i, j), col] = 1
                V26[idx_v(j, k), col] = 1
                W26[idx_w(k, i), col] = 1
                col += 1
                
    # Reshape for optimization
    U26_r = U26.reshape(n, m, R_start)
    V26_r = V26.reshape(m, p, R_start)
    W26_r = W26.reshape(p, n, R_start)
    
    num_params = R * (n * m + m * p + p * n)
    
    def tensor_error(params):
        U = params[0 : R * n * m].reshape(n, m, R)
        V = params[R * n * m : 2 * R * n * m].reshape(m, p, R)
        W = params[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        T_tgt = np.ones((n, m, p))
        return (T_tgt - T_rec).flatten()
        
    def round_and_verify(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_tgt = np.ones((n, m, p))
        return np.allclose(T_rec, T_tgt, atol=1e-12)

    border_indices = list(range(7, 26))
    
    np.random.seed(42)
    max_attempts = 20
    
    for _ in range(max_attempts):
        if time.time() > deadline - 10:
            break
            
        drop_indices = np.random.choice(border_indices, 3, replace=False)
        keep_indices = [i for i in range(R_start) if i not in drop_indices]
        
        U_init = U26_r[:, :, keep_indices].flatten()
        V_init = V26_r[:, :, keep_indices].flatten()
        W_init = W26_r[:, :, keep_indices].flatten()
        
        params_init = np.concatenate([U_init, V_init, W_init])
        
        try:
            res = least_squares(
                tensor_error,
                params_init,
                method='lm',
                max_nfev=300,
                x_scale='jac',
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
        except Exception:
            continue
            
        if res.cost < 1e-10:
            U_opt = res.x[0 : R * n * m].reshape(n, m, R)
            V_opt = res.x[R * n * m : 2 * R * n * m].reshape(m, p, R)
            W_opt = res.x[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
            
            if round_and_verify(U_opt, V_opt, W_opt):
                U_final = np.round(U_opt * 2) / 2
                V_final = np.round(V_opt * 2) / 2
                W_final = np.round(W_opt * 2) / 2
                U_flat = U_final.reshape(n * m, R)
                V_flat = V_final.reshape(m * p, R)
                W_flat = W_final.reshape(p * n, R)
                return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
                
    return None
