def _search_rank(n, m, p, r, time_limit):
    start = time.time()
    U = np.random.randint(-2, 3, (n*m, r))
    V = np.random.randint(-2, 3, (m*p, r))
    W = np.random.randint(-2, 3, (p*n, r))
    
    # Initial error
    err = _calc_error(U, V, W, n, m, p)
    
    while time.time() - start < time_limit:
        # Random move
        mat = np.random.choice(3)
        if mat == 0:
            idx = np.random.randint(n*m)
            col = np.random.randint(r)
            old = U[idx, col]
            U[idx, col] += np.random.choice([-1, -0.5, 0.5, 1])
        elif mat == 1:
            idx = np.random.randint(m*p)
            col = np.random.randint(r)
            old = V[idx, col]
            V[idx, col] += np.random.choice([-1, -0.5, 0.5, 1])
        else:
            idx = np.random.randint(p*n)
            col = np.random.randint(r)
            old = W[idx, col]
            W[idx, col] += np.random.choice([-1, -0.5, 0.5, 1])
            
        new_err = _calc_error(U, V, W, n, m, p)
        
        if new_err < err or np.random.random() < np.exp((err - new_err) / 0.1): # SA accept
            err = new_err
            if err < 1e-9:
                # Verify exact
                if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U, V, W
        else:
            # Revert
            if mat == 0: U[idx, col] = old
            elif mat == 1: V[idx, col] = old
            else: W[idx, col] = old
            
    return None
