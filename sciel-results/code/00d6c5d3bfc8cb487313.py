import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Initialize empty matrices (list of columns/terms)
    # It is easier to build columns as we add terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m
        v_vec: list of length m*p
        w_vec: list of length p*n
        """
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    def add_strassen_block(i_off, j_off, k_off):
        """
        Adds a rank-7 decomposition for a 2x2x2 block
        starting at global indices i_off, j_off, k_off.
        Local i, j, k in {0, 1}.
        """
        # Define local 2x2 Strassen vectors
        # u: (i*m + j) -> local index 0..3. Global index (i_off+i)*m + (j_off+j)
        # v: (j*p + k) -> local index 0..5? No, for 2x2x2, p=2.
        # Wait, the Strassen block assumes p=2 locally.
        
        # Strassen Terms (u, v, w) for 2x2x2
        # Index mapping local:
        # U_idx = i*2 + j
        # V_idx = j*2 + k
        # W_idx = k*2 + i
        
        # T1: P1
        u1 = [0,0,0,0]
        u1[0] = 1; u1[3] = 1 # (0,0), (1,1)
        v1 = [0,0,0,0]
        v1[0] = 1; v1[3] = 1 # (0,0), (1,1)
        w1 = [0,0,0,0]
        w1[0] = 1; w1[3] = 1 # (0,0), (1,1)
        
        # T2: P2
        u2 = [0,0,0,0]
        u2[2] = 1; u2[3] = 1 # (1,0), (1,1)
        v2 = [0,0,0,0]
        v2[0] = 1 # (0,0)
        w2 = [0,0,0,0]
        w2[2] = 1 # C(1,0) -> k=1, i=0 -> idx 2
        w2[3] = -1 # C(1,1) -> k=1, i=1 -> idx 3
        
        # T3: P3
        u3 = [0,0,0,0]
        u3[0] = 1 # (0,0)
        v3 = [0,0,0,0]
        v3[1] = 1; v3[3] = -1 # (0,1), (1,1)
        w3 = [0,0,0,0]
        w3[1] = 1 # C(0,1) -> k=0, i=1 -> idx 1
        w3[3] = 1 # C(1,1) -> idx 3
        
        # T4: P4
        u4 = [0,0,0,0]
        u4[3] = 1 # (1,1)
        v4 = [0,0,0,0]
        v4[0] = -1; v4[2] = 1 # (0,0), (1,0)
        w4 = [0,0,0,0]
        w4[0] = 1 # C(0,0)
        w4[2] = 1 # C(1,0)
        
        # T5: P5
        u5 = [0,0,0,0]
        u5[0] = 1; u5[1] = 1 # (0,0), (0,1)
        v5 = [0,0,0,0]
        v5[3] = 1 # (1,1)
        w5 = [0,0,0,0]
        w5[0] = -1 # C(0,0)
        w5[1] = 1 # C(0,1)
        
        # T6: P6
        u6 = [0,0,0,0]
        u6[0] = -1; u6[2] = 1 # (0,0), (1,0)
        v6 = [0,0,0,0]
        v6[0] = 1; v6[1] = 1 # (0,0), (0,1)
        w6 = [0,0,0,0]
        w6[3] = 1 # C(1,1)
        
        # T7: P7
        u7 = [0,0,0,0]
        u7[1] = 1; u7[3] = -1 # (0,1), (1,1)
        v7 = [0,0,0,0]
        v7[2] = 1; v7[3] = 1 # (1,0), (1,1)
        w7 = [0,0,0,0]
        w7[0] = 1 # C(0,0)
        
        local_terms = [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]
        
        # Expand to global size
        for u_loc, v_loc, w_loc in local_terms:
            u_global = [0] * (n*m)
            v_global = [0] * (m*p)
            w_global = [0] * (p*n)
            
            # Map U
            for i in range(2):
                for j in range(2):
                    val = u_loc[i*2 + j]
                    if val != 0:
                        g_i = i_off + i
                        g_j = j_off + j
                        u_global[g_i*m + g_j] = val
                        
            # Map V
            for j in range(2):
                for k in range(2):
                    val = v_loc[j*2 + k]
                    if val != 0:
                        g_j = j_off + j
                        g_k = k_off + k
                        v_global[g_j*p + g_k] = val
                        
            # Map W
            for k in range(2):
                for i in range(2):
                    val = w_loc[k*2 + i]
                    if val != 0:
                        g_k = k_off + k
                        g_i = i_off + i
                        w_global[g_k*n + g_i] = val
                        
            add_term(u_global, v_global, w_global)
            
    def add_schoolbook(i, j, k):
        u_global = [0] * (n*m)
        v_global = [0] * (m*p)
        w_global = [0] * (p*n)
        
        u_global[i*m + j] = 1
        v_global[j*p + k] = 1
        w_global[k*n + i] = 1
        
        add_term(u_global, v_global, w_global)

    # 1. J in {0, 1}
    # i in {0, 1}, j in {0, 1} -> Strassen on k-blocks
    for k_off in [0, 2, 4]:
        add_strassen_block(i_off=0, j_off=0, k_off=k_off)
        
    # i in {2}, j in {0, 1} -> Schoolbook
    for j in [0, 1]:
        for k in range(p):
            add_schoolbook(i=2, j=j, k=k)

    # 2. J in {2, 3}
    # i in {0, 1}, j in {2, 3} -> Strassen on k-blocks
    for k_off in [0, 2, 4]:
        add_strassen_block(i_off=0, j_off=2, k_off=k_off)
        
    # i in {2}, j in {2, 3} -> Schoolbook
    for j in [2, 3]:
        for k in range(p):
            add_schoolbook(i=2, j=j, k=k)

    # Convert list of columns to list of rows for U, V, W
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for r in range(rank):
        u_vec = U_terms[r]
        v_vec = V_terms[r]
        w_vec = W_terms[r]
        
        for idx in range(n*m):
            if u_vec[idx] != 0:
                U[idx][r] = u_vec[idx]
        for idx in range(m*p):
            if v_vec[idx] != 0:
                V[idx][r] = v_vec[idx]
        for idx in range(p*n):
            if w_vec[idx] != 0:
                W[idx][r] = w_vec[idx]
                
    return {"U": U, "V": V, "W": W}
