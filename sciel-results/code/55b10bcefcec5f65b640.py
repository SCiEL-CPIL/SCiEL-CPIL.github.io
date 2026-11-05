def construct():
    import numpy as np
    import time
    
    n, m, p = 4, 5, 5
    
    # Winograd 2x2x2 Rank 7 Scheme
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    # Note: W index in prompt is k*n+i. For 2x2, n=2, so k*2+i.
    
    U2 = np.array([
        [1, 1, 0, 0, 1, 0, 0], # i=0, j=0
        [0, 0, 1, 1, 0, 1, 0], # i=0, j=1
        [1, -1, 0, 0, 1, 0, 0],# i=1, j=0
        [0, 0, 1, -1, 0, 1, 0] # i=1, j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 0, 1], # j=0, k=0
        [0, 0, 1, 1, 0, 0, 1], # j=0, k=1
        [1, -1, 0, 0, 0, 0, 1],# j=1, k=0
        [0, 0, 1, -1, 0, 0, 1] # j=1, k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 1, 0, 1, 1, 0], # k=0, i=0
        [0, 1, 0, 1, 1, 1, 0], # k=0, i=1
        [1, 0, 1, 0, 0, 0, 1], # k=1, i=0
        [0, 1, 0, 1, 0, 0, 1]  # k=1, i=1
    ], dtype=float)
    
    # Verify 2x2x2
    # C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) (sum_j' V[j',k,t] B[j',k]) W[k,i,t]
    # This specific scheme needs to be checked.
    # I will use a generic schoolbook for 2x2x2 if verification fails, but rank 7 is known.
    
    # Partition
    row_splits = [2, 2]
    mid_splits = [2, 2, 1]
    col_splits = [2, 2, 1]
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    r_offset = 0
    for r_size in row_splits:
        m_offset = 0
        for m_size in mid_splits:
            k_offset = 0
            for k_size in col_splits:
                if r_size == 2 and m_size == 2 and k_size == 2:
                    # Use Winograd
                    for t in range(7):
                        u_vec = np.zeros(20)
                        v_vec = np.zeros(25)
                        w_vec = np.zeros(20)
                        
                        for dr in range(2):
                            for dm in range(2):
                                u_idx = (r_offset + dr) * m + (m_offset + dm)
                                u_vec[u_idx] = U2[dr*2+dm, t]
                                
                        for dm in range(2):
                            for dk in range(2):
                                v_idx = (m_offset + dm) * p + (k_offset + dk)
                                v_vec[v_idx] = V2[dm*2+dk, t]
                                
                        for dk in range(2):
                            for dr in range(2):
                                w_idx = (k_offset + dk) * n + (r_offset + dr)
                                w_vec[w_idx] = W2[dk*2+dr, t]
                                
                        terms_U.append(u_vec)
                        terms_V.append(v_vec)
                        terms_W.append(w_vec)
                else:
                    # Schoolbook
                    for dr in range(r_size):
                        for dm in range(m_size):
                            for dk in range(k_size):
                                u_vec = np.zeros(20)
                                v_vec = np.zeros(25)
                                w_vec = np.zeros(20)
                                
                                u_idx = (r_offset + dr) * m + (m_offset + dm)
                                u_vec[u_idx] = 1
                                
                                v_idx = (m_offset + dm) * p + (k_offset + dk)
                                v_vec[v_idx] = 1
                                
                                w_idx = (k_offset + dk) * n + (r_offset + dr)
                                w_vec[w_idx] = 1
                                
                                terms_U.append(u_vec)
                                terms_V.append(v_vec)
                                terms_W.append(w_vec)
                k_offset += k_size
            m_offset += m_size
        r_offset += r_size
        
    rank = len(terms_U)
    U = np.array(terms_U).T
    V = np.array(terms_V).T
    W = np.array(terms_W).T
    
    # Verify
    np.random.seed(42)
    A = np.random.randn(4, 5)
    B = np.random.randn(5, 5)
    C_expected = A @ B
    C_computed = np.zeros((4, 5))
    
    for t in range(rank):
        u_prod = np.dot(U[:, t], A.flatten())
        v_prod = np.dot(V[:, t], B.flatten())
        w_vec = W[:, t]
        C_computed += u_prod * v_prod * w_vec.reshape(5, 4).T
        
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook if Winograd scheme is wrong
        terms_U = []
        terms_V = []
        terms_W = []
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    u_vec = np.zeros(20)
                    v_vec = np.zeros(25)
                    w_vec = np.zeros(20)
                    u_vec[i*5+j] = 1
                    v_vec[j*5+k] = 1
                    w_vec[k*4+i] = 1
                    terms_U.append(u_vec)
                    terms_V.append(v_vec)
                    terms_W.append(w_vec)
        U = np.array(terms_U).T
        V = np.array(terms_V).T
        W = np.array(terms_W).T
        rank = len(terms_U)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
