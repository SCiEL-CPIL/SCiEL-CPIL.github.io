def _local_search_half_int(U, V, W, n, m, p, R, max_iters=100):
    """Try to reduce error by flipping individual entries to nearest half-integers."""
    nm, mp, pn = n * m, m * p, p * n
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    def get_error():
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    current_err = get_error()
    for _ in range(max_iters):
        improved = False
        # Try flipping a random entry in U, V, or W
        # To be faster, we can precompute contributions, but for small R, direct calc is ok.
        
        # Randomly pick a matrix and an entry
        mat = np.random.choice(['U', 'V', 'W'])
        if mat == 'U':
            idx = np.random.randint(nm * R)
            i, r = divmod(idx, R)
            old_val = U[i, r]
            # Try nearest half-integer
            new_val = np.round(old_val * 2) / 2
            if new_val != old_val:
                U[i, r] = new_val
                new_err = get_error()
                if new_err < current_err - 1e-9:
                    current_err = new_err
                    improved = True
                else:
                    U[i, r] = old_val
        elif mat == 'V':
            idx = np.random.randint(mp * R)
            i, r = divmod(idx, R)
            old_val = V[i, r]
            new_val = np.round(old_val * 2) / 2
            if new_val != old_val:
                V[i, r] = new_val
                new_err = get_error()
                if new_err < current_err - 1e-9:
                    current_err = new_err
                    improved = True
                else:
                    V[i, r] = old_val
        else:
            idx = np.random.randint(pn * R)
            i, r = divmod(idx, R)
            old_val = W[i, r]
            new_val = np.round(old_val * 2) / 2
            if new_val != old_val:
                W[i, r] = new_val
                new_err = get_error()
                if new_err < current_err - 1e-9:
                    current_err = new_err
                    improved = True
                else:
                    W[i, r] = old_val
                    
        if not improved:
            break
            
    if current_err < 1e-9:
        return U, V, W
    return None

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower ranks
    for R in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        result = _try_optimization(n, m, p, R, deadline, num_trials=3)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = R
            
    return {"U": best_U, "V": best_V, "W": best_W}
