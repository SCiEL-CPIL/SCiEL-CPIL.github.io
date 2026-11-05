def construct_216():
    rank = 216
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    col = 0
    for i in range(3):  # block row of A
        for j in range(3):  # block col of A = block row of B
            for k in range(3):  # block col of B
                for a in range(2):  # row within 2x2
                    for b in range(2):  # col within 2x2 (A) / row within 2x2 (B)
                        for c in range(2):  # col within 2x2 (B)
                            # A entry: global (2i+a, 2j+b)
                            u_idx = (2*i + a) * 6 + (2*j + b)
                            # B entry: global (2j+b, 2k+c)
                            v_idx = (2*j + b) * 6 + (2*k + c)
                            # C entry: global (2i+a, 2k+c), W index: (2k+c)*6 + (2i+a)
                            w_idx = (2*k + c) * 6 + (2*i + a)
                            
                            U[u_idx, col] = 1.0
                            V[v_idx, col] = 1.0
                            W[w_idx, col] = 1.0
                            col += 1
    return U, V, W
