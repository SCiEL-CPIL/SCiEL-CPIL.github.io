import numpy as np

def construct():
    n, m, p = 2, 2, 14
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # We define the 7 terms (u, v, w)
    # u is A-coeffs, v is B-coeffs, w is C-coeffs
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # C00 += 1, C11 += 1
    # Wait, C11 = P1 - P2 + P3 + P6. So P1 contributes +1 to C11.
    # C00 = P1 + P4 - P5 + P7. So P1 contributes +1 to C00.
    
    # Let's list (u, v, w) for each P_k
    # u, v, w are lists of 4 elements
    
    terms = []
    
    # P1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C00, C11
    terms.append((u1, v1, w1))
    
    # P2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C10, -C11
    terms.append((u2, v2, w2))
    
    # P3
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 1] # C01, C11
    terms.append((u3, v3, w3))
    
    # P4
    u4 = [0, 0, 0, 1]
    v4 = [-1, 1, 0, 0] # B01 - B00
    w4 = [1, 0, 1, 0] # C00, C10
    terms.append((u4, v4, w4))
    
    # P5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # -C00, C01
    terms.append((u5, v5, w5))
    
    # P6
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C11
    terms.append((u6, v6, w6))
    
    # P7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C00
    terms.append((u7, v7, w7))
    
    # Verification logic (commented out for final submission)
    # A = np.array([[1,2],[3,4]])
    # B = np.array([[5,6],[7,8]])
    # C_true = A @ B
    # C_strassen = np.zeros((2,2))
    # for u,v,w in terms:
    #     ua = np.dot(u, A.flatten())
    #     vb = np.dot(v, B.flatten())
    #     C_strassen += ua * vb * np.array(w).reshape(2,2)
    # assert np.allclose(C_true, C_strassen)
    
    # Fill U, V, W
    for t in range(7):
        for k in range(7):
            col = t * 7 + k
            u, v, w = terms[k]
            
            # U: A indices 0..3
            for i in range(4):
                if u[i] != 0:
                    U[i][col] = u[i]
            
            # V: B indices for block t
            # Local 0: b0,2t -> idx 2t
            # Local 1: b0,2t+1 -> idx 2t+1
            # Local 2: b1,2t -> idx 14+2t
            # Local 3: b1,2t+1 -> idx 14+2t+1
            v_indices = [2*t, 2*t+1, 14+2*t, 14+2*t+1]
            for i in range(4):
                if v[i] != 0:
                    V[v_indices[i]][col] = v[i]
            
            # W: C indices for block t
            # Local 0: c0,2t -> idx 4t
            # Local 1: c0,2t+1 -> idx 4t+2
            # Local 2: c1,2t -> idx 4t+1
            # Local 3: c1,2t+1 -> idx 4t+3
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for i in range(4):
                if w[i] != 0:
                    W[w_indices[i]][col] = w[i]
    
    return {"U": U, "V": V, "W": W}
