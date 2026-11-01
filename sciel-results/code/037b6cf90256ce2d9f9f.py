def construct():
    n, m, p = 2, 3, 5
    
    # Total rank
    rank_sp1 = 17
    rank_sp2 = 11
    total_rank = rank_sp1 + rank_sp2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(u_off, v_off, w_off):
        nonlocal col
        # 7 terms
        # 1. m1: (a+d)(e+h)
        U[u_off + 0][col] += 1; U[u_off + 3][col] += 1
        V[v_off + 0][col] += 1; V[v_off + 2][col] += 1
        W[w_off + 0][col] += 1; W[w_off + 1][col] += 1 # W index: j*n + i. C_00, C_11.
        # Wait, W index for C_00 is 0*2+0=0. C_11 is 1*2+1=3.
        # My W offset logic needs to be careful.
        # W is flattened p*n.
        # For a 2x2 C block, indices are (i,j) with i in 0,1 and j in 0,1.
        # W index = j*n + i.
        # C_00 -> j=0, i=0 -> 0
        # C_01 -> j=1, i=0 -> 2
        # C_10 -> j=0, i=1 -> 1
        # C_11 -> j=1, i=1 -> 3
        pass # Implementation below

    # We will implement the specific additions inline or via a helper that takes base indices.
    
    return {"U": U, "V": V, "W": W}
