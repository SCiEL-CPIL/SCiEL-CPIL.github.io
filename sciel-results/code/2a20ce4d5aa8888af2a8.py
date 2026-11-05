import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Split: n = 1+2, m = 2+2, p = 2+3
    n1, n2 = 1, 2
    m1, m2 = 2, 2
    p1, p2 = 2, 3
    
    # Get decompositions for each sub-block
    def schoolbook_decomp(n_, m_, p_):
        """Return U, V, W for schoolbook n_ x m_ times m_ x p_"""
        rank = n_ * m_ * p_
        U = [[0]*rank for _ in range(n_*m_)]
        V = [[0]*rank for _ in range(m_*p_)]
        W = [[0]*rank for _ in range(p_*n_)]
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i*m_+j][col] = 1
                    V[j*p_+k][col] = 1
                    W[k*n_+i][col] = 1
                    col += 1
        return U, V, W
    
    def strassen_222():
        """Return U, V, W for Strassen 2x2 x 2x2, rank 7"""
        # Input A: indices 0=a, 1=b, 2=c, 3=d
        # Input B: indices 0=e, 1=f, 2=g, 3=h
        # Output C: indices 0=C11, 1=C12, 2=C21, 3=C22
        # 
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        U = [
            [1, 0, 1, 0, 1, 0, 0],  # a: m1, m3, m5
            [0, 0, 0, 0, 1, 0, 1],  # b: m5, m7
            [0, 1, 0, 0, 0, 1, 0],  # c: m2, m6
            [1, 1, 0, 1, 0, 0, 0],  # d: m1, m2, m4
        ]
        V = [
            [1, 1, 0, 0, 0, 1, 0],  # e: m1, m2, m6
            [0, 0, 1, 0, 0, 1, 0],  # f: m3, m6
            [0, 0, 0, 1, 0, 0, 1],  # g: m4, m7
            [1, 0, -1, 0, 1, 0, 1], # h: m1, m3, m5, m7
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1],  # C11: m1, m4, -m5, m7
            [0, 0, 1, 0, 1, 0, 0],   # C12: m3, m5
            [0, 1, 0, 1, 0, 0, 0],   # C21: m2, m4
            [1, -1, 1, 0, 0, 1, 0],  # C22: m1, -m2, m3, m6
        ]
        return U, V, W
    
    # Build the 8 sub-block decompositions
    blocks = {}
    # Block (ib, jb, kb): ib in {0,1}, jb in {0,1}, kb in {0,1}
    # ib=0 -> n1=1, ib=1 -> n2=2
    # jb=0 -> m1=2, jb=1 -> m2=2
    # kb=0 -> p1=2, kb=1 -> p2=3
    
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                n_ = n1 if ib == 0 else n2
                m_ = m1 if jb == 0 else m2
                p_ = p1 if kb == 0 else p2
                if n_ == 2 and m_ == 2 and p_ == 2:
                    blocks[(ib,jb,kb)] = strassen_222()
                else:
                    blocks[(ib,jb,kb)] = schoolbook_decomp(n_, m_, p_)
    
    # Compute total rank
    total_rank = 0
    for key in sorted(blocks.keys()):
        total_rank += len(blocks[key][0][0])
    
    # Allocate global matrices
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Fill in
    rank_offset = 0
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                Ub, Vb, Wb = blocks[(ib,jb,kb)]
                r = len(Ub[0])
                
                # Row offsets in global matrices
                # U: row index i*m + j, where i is in [0,n), j is in [0,m)
                # For block (ib, jb, kb):
                #   i ranges: if ib==0, i in [0,1); if ib==1, i in [1,3)
                #   j ranges: if jb==0, j in [0,2); if jb==1, j in [2,4)
                i_start = 0 if ib == 0 else n1
                i_end = n1 if ib == 0 else n
                j_start = 0 if jb == 0 else m1
                j_end = m1 if jb == 0 else m
                
                # V: row index j*p + k
                k_start = 0 if kb == 0 else p1
                k_end = p1 if kb == 0 else p
                
                # W: row index k*n + i
                # k ranges: if kb==0, k in [0,2); if kb==1, k in [2,5)
                # i ranges: if ib==0, i in [0,1); if ib==1, i in [1,3)
                
                n_ = n1 if ib == 0 else n2
                m_ = m1 if jb == 0 else m2
                p_ = p1 if kb == 0 else p2
                
                for local_i in range(n_):
                    for local_j in range(m_):
                        global_i = i_start + local_i
                        global_j = j_start + local_j
                        u_row = global_i * m + global_j
                        for r_idx in range(r):
                            U[u_row][rank_offset + r_idx] = Ub[local_i*m_+local_j][r_idx]
                
                for local_j in range(m_):
                    for local_k in range(p_):
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        v_row = global_j * p + global_k
                        for r_idx in range(r):
                            V[v_row][rank_offset + r_idx] = Vb[local_j*p_+local_k][r_idx]
                
                for local_k in range(p_):
                    for local_i in range(n_):
                        global_k = k_start + local_k
                        global_i = i_start + local_i
                        w_row = global_k * n + global_i
                        for r_idx in range(r):
                            W[w_row][rank_offset + r_idx] = Wb[local_k*n_+local_i][r_idx]
                
                rank_offset += r
    
    return {"U": U, "V": V, "W": W}
