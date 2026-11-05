    # Part 3: 4x1x5 (k=4, j in 0..4) - Single rank-1 term
    u_vec = np.zeros(n * m)
    v_vec = np.zeros(m * p)
    w_vec = np.zeros(p * n)
    
    for i in range(n):
        u_vec[i * m + 4] = 1.0
        
    for j in range(p):
        v_vec[4 * p + j] = 1.0
        
    for j in range(p):
        for i in range(n):
            w_vec[j * n + i] = 1.0
            
    add_rank1(u_vec, v_vec, w_vec)
