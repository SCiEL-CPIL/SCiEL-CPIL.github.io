def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Directly assume y' = A*y + b based on task description
    # 1. Compute b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # 2. Compute A columns: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float) - b
        
    # 3. Use augmented matrix exponential for exact solution
    # [y; 1]' = [A, b; 0, 0] [y; 1]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    
    M_exp = expm(M * dt)
    
    y_aug_init = np.zeros(n + 1)
    y_aug_init[:n] = y0
    y_aug_init[n] = 1.0
    
    y_aug_end = M_exp @ y_aug_init
    return y_aug_end[:n]
