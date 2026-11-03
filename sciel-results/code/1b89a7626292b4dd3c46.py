"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_indices, v_indices, w_indices):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in u_indices:
            u_vec[idx] += val
        for idx, val in v_indices:
            v_vec[idx] += val
        for idx, val in w_indices:
            w_vec[idx] += val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # 1. Add Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            i0 = 0
            
            # M1 = (A00 + A11)(B00 + B11)
            u1 = [(i0*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
            v1 = [(j0*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
            w1 = [(k0*n + i0, 1), ((k0+1)*n + (i0+1), 1)]
            add_col(u1, v1, w1)
            
            # M2 = (A10 + A11)B00
            u2 = [((i0+1)*m + j0, 1), ((i0+1)*m + (j0+1), 1)]
            v2 = [(j0*p + k0, 1)]
            w2 = [(k0*n + (i0+1), 1)]
            add_col(u2, v2, w2)
            
            # M3 = A00(B10 - B11)
            u3 = [(i0*m + j0, 1)]
            v3 = [((j0+1)*p + k0, 1), ((j0+1)*p + (k0+1), -1)]
            w3 = [(k0*n + i0, 1), ((k0+1)*n + i0, 1)]
            add_col(u3, v3, w3)
            
            # M4 = A11(B01 - B00)
            u4 = [((i0+1)*m + (j0+1), 1)]
            v4 = [(j0*p + (k0+1), 1), (j0*p + k0, -1)]
            w4 = [(k0*n + i0, 1), (k0*n + (i0+1), 1)]
            add_col(u4, v4, w4)
            
            # M5 = (A00 + A01)B11
            u5 = [(i0*m + j0, 1), (i0*m + (j0+1), 1)]
            v5 = [((j0+1)*p + (k0+1), 1)]
            w5 = [(k0*n + i0, -1), ((k0+1)*n + i0, 1)]
            add_col(u5, v5, w5)
            
            # M6 = (A10 - A00)(B00 + B01)
            u6 = [((i0+1)*m + j0, 1), (i0*m + j0, -1)]
            v6 = [(j0*p + k0, 1), (j0*p + (k0+1), 1)]
            w6 = [((k0+1)*n + (i0+1), 1)]
            add_col(u6, v6, w6)
            
            # M7 = (A01 - A11)(B10 + B11)
            u7 = [(i0*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
            v7 = [((j0+1)*p + k0, 1), ((j0+1)*p + (k0+1), 1)]
            w7 = [(k0*n + i0, 1)]
            add_col(u7, v7, w7)
            
    # 2. Add Schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3, 4, 5]:
                    j0 = j if j % 2 == 0 else j - 1
                    k0 = k if k % 2 == 0 else k - 1
                    if j0 in [0, 2] and k0 in [0, 2, 4]:
                        covered = True
                
                if not covered:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    u_vec = [0] * (n * m)
                    v_vec = [0] * (m * p)
                    w_vec = [0] * (p * n)
                    
                    u_vec[u_idx] = 1
                    v_vec[v_idx] = 1
                    w_vec[w_idx] = 1
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                    
    rank = len(U_cols)
    U = [list(row) for row in zip(*U_cols)]
    V = [list(row) for row in zip(*V_cols)]
    W = [list(row) for row in zip(*W_cols)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
