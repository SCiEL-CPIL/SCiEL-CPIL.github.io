import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Partitions
    I_blocks = [[0, 1], [2, 3]]
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4]]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_schoolbook_block(I_idx, J_idx, K_idx):
        for i in I_idx:
            for j in J_idx:
                for k in K_idx:
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    
                    U_terms.append(u)
                    V_terms.append(v)
                    W_terms.append(w)
                    
    def add_strassen_block(I_idx, J_idx, K_idx):
        if len(I_idx) != 2 or len(J_idx) != 2 or len(K_idx) != 2:
            add_schoolbook_block(I_idx, J_idx, K_idx)
            return
            
        i0, i1 = I_idx
        j0, j1 = J_idx
        k0, k1 = K_idx
        
        def make_term(u_coeffs, v_coeffs, w_coeffs):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            for (ii, jj), val in u_coeffs.items():
                u[ii * m + jj] = val
            for (jj, kk), val in v_coeffs.items():
                v[jj * p + kk] = val
            for (ii, kk), val in w_coeffs.items():
                w[kk * n + ii] = val
                
            return u, v, w
        
        # M1
        u1 = {(i0,j0): 1, (i1,j1): 1}
        v1 = {(j0,k0): 1, (j1,k1): 1}
        w1 = {(i0,k0): 1, (i1,k1): 1}
        U_terms.append(make_term(u1, v1, w1)[0])
        V_terms.append(make_term(u1, v1, w1)[1])
        W_terms.append(make_term(u1, v1, w1)[2])
        
        # M2
        u2 = {(i1,j0): 1, (i1,j1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(i1,k0): 1, (i1,k1): -1}
        U_terms.append(make_term(u2, v2, w2)[0])
        V_terms.append(make_term(u2, v2, w2)[1])
        W_terms.append(make_term(u2, v2, w2)[2])
        
        # M3
        u3 = {(i0,j0): 1}
        v3 = {(j0,k1): 1, (j1,k1): -1}
        w3 = {(i0,k1): 1, (i1,k1): 1}
        U_terms.append(make_term(u3, v3, w3)[0])
        V_terms.append(make_term(u3, v3, w3)[1])
        W_terms.append(make_term(u3, v3, w3)[2])
        
        # M4
        u4 = {(i1,j1): 1}
        v4 = {(j1,k0): 1, (j0,k0): -1}
        w4 = {(i0,k0): 1, (i1,k0): 1}
        U_terms.append(make_term(u4, v4, w4)[0])
        V_terms.append(make_term(u4, v4, w4)[1])
        W_terms.append(make_term(u4, v4, w4)[2])
        
        # M5
        u5 = {(i0,j0): 1, (i0,j1): 1}
        v5 = {(j1,k1): 1}
        w5 = {(i0,k0): -1, (i0,k1): 1}
        U_terms.append(make_term(u5, v5, w5)[0])
        V_terms.append(make_term(u5, v5, w5)[1])
        W_terms.append(make_term(u5, v5, w5)[2])
        
        # M6
        u6 = {(i1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j0,k1): 1}
        w6 = {(i1,k1): 1}
        U_terms.append(make_term(u6, v6, w6)[0])
        V_terms.append(make_term(u6, v6, w6)[1])
        W_terms.append(make_term(u6, v6, w6)[2])
        
        # M7
        u7 = {(i0,j1): 1, (i1,j1): -1}
        v7 = {(j1,k0): 1, (j1,k1): 1}
        w7 = {(i0,k0): 1}
        U_terms.append(make_term(u7, v7, w7)[0])
        V_terms.append(make_term(u7, v7, w7)[1])
        W_terms.append(make_term(u7, v7, w7)[2])

    for I_idx in I_blocks:
        for J_idx in J_blocks:
            for K_idx in K_blocks:
                if len(I_idx) == 2 and len(J_idx) == 2 and len(K_idx) == 2:
                    add_strassen_block(I_idx, J_idx, K_idx)
                else:
                    add_schoolbook_block(I_idx, J_idx, K_idx)
                    
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    return {"U": U, "V": V, "W": W}
