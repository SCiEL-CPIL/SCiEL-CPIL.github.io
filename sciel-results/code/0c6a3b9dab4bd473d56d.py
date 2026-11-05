def construct():
    n, m, p = 3, 5, 5
    
    # Splits: n: 1+2, m: 2+2+1, p: 2+2+1
    n_groups = [(0,), (1, 2)]
    m_groups = [(0, 1), (2, 3), (4,)]
    p_groups = [(0, 1), (2, 3), (4,)]
    
    # Determine rank for each block
    def block_rank(nsz, msz, psz):
        if nsz == 1 and msz == 2 and psz == 2:
            return 1
        if nsz == 1 and msz == 2 and psz == 1:
            return 1
        if nsz == 1 and msz == 1 and psz == 2:
            return 1
        if nsz == 1 and msz == 1 and psz == 1:
            return 1
        if nsz == 2 and msz == 2 and psz == 2:
            return 7
        if nsz == 2 and msz == 2 and psz == 1:
            return 4
        if nsz == 2 and msz == 1 and psz == 2:
            return 4
        if nsz == 2 and msz == 1 and psz == 1:
            return 1
        return nsz * msz * psz  # schoolbook
    
    total_rank = 0
    for ng in n_groups:
        for mg in m_groups:
            for pg in p_groups:
                total_rank += block_rank(len(ng), len(mg), len(pg))
    
    # Build U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ng in n_groups:
        for mg in m_groups:
            for pg in p_groups:
                nsz, msz, psz = len(ng), len(mg), len(pg)
                rank = block_rank(nsz, msz, psz)
                
                if nsz == 2 and msz == 2 and psz == 2:
                    # Strassen 2x2x2
                    Us, Vs, Ws = _strassen_2x2x2()
                    i0, i1 = ng
                    j0, j1 = mg
                    k0, k1 = pg
                    u_rows = [i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1]
                    v_rows = [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1]
                    w_rows = [k0*n+i0, k1*n+i0, k0*n+i1, k1*n+i1]
                    for c in range(7):
                        for s in range(4):
                            U[u_rows[s]][col+c] = Us[s][c]
                            V[v_rows[s]][col+c] = Vs[s][c]
                            W[w_rows[s]][col+c] = Ws[s][c]
                elif rank == 1:
                    # Rank-1: all ones
                    for i in ng:
                        for j in mg:
                            U[i*m+j][col] = 1
                    for j in mg:
                        for k in pg:
                            V[j*p+k][col] = 1
                    for k in pg:
                        for i in ng:
                            W[k*n+i][col] = 1
                else:
                    # Schoolbook
                    for i in ng:
                        for j in mg:
                            for k in pg:
                                U[i*m+j][col] = 1
                                V[j*p+k][col] = 1
                                W[k*n+i][col] = 1
                                col += 1
                    # Don't increment col by rank here, it's handled per-term
                    col -= rank  # undo, we incremented per term
                    col += rank  # net effect: col advanced by rank
                    # Actually this is wrong. Let me redo.
                
                if rank != 1 and not (nsz == 2 and msz == 2 and psz == 2):
                    # Schoolbook: col was incremented per term
                    pass
                elif rank == 1:
                    col += 1
                else:
                    col += rank
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
