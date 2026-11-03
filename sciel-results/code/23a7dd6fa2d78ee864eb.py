def _local_reduce(U, V, W, n, m, p, max_iters=50, budget_s=100.0):
    """Local column-removal search with randomization."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if _time_left() < 10.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        cols = list(range(R))
        rng.shuffle(cols)
        
        for c in cols:
            if _time_left() < 10.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            # Use fast vectorized check with tight tolerance
            if _verify_np(U2, V2, W2, n, m, p, 1e-9):
                # Optional: exact check here if needed, but _verify_np is likely sufficient for integer/half-integer inputs
                U, V, W = U2, V2, W2
                R -= 1
                found = True
                break
        if not found:
            break
    return U, V, W
