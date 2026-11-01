def _als_from_init(U_init, V_init, W_init, n, m, p, rank, iterations=500):
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    
    prev_error = float('inf')
    for it in range(iterations):
        # ... same updates ...
        
    # snap and verify
