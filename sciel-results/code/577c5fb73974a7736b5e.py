import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a sub-decomposition
    # sub_n, sub_m, sub_p: dimensions of the sub-problem
    # offset_i, offset_j, offset_k: starting indices for i, j, k in the global problem
    # sub_U, sub_V, sub_W: local matrices
    # col_offset: starting column index in the global U, V, W
    def add_subproblem(sub_n, sub_m, sub_p, offset_i, offset_j, offset_k, sub_U, sub_V, sub_W, col_offset):
        for c in range(sub_U.shape[1]):
            for i in range(sub_n):
                for j in range(sub_m):
                    if sub_U[i * sub_m + j, c] != 0:
                        U[(offset_i + i) * m + (offset_j + j), col_offset + c] += sub_U[i * sub_m + j, c]
            for j in range(sub_m):
                for k in range(sub_p):
                    if sub_V[j * sub_p + k, c] != 0:
                        V[(offset_j + j) * p + (offset_k + k), col_offset + c] += sub_V[j * sub_p + k, c]
            for k in range(sub_p):
                for i in range(sub_n):
                    if sub_W[k * sub_n + i, c] != 0:
                        W[(offset_k + k) * n + (offset_i + i), col_offset + c] += sub_W[k * sub_n + i, c]

    # 1. Strassen 2x2x2
    def strassen_222():
        r = 7
        u = np.zeros((4, r))
        v = np.zeros((4, r))
        w = np.zeros((4, r))
        
        # M1 = A00 * (B00 + B11)
        u[0, 0] = 1
        v[0, 0] = 1
        v[3, 0] = 1
        w[0, 0] = 1
        w[1, 0] = 1 # C00 gets M1, C11 gets M1? No.
        # Let's check Strassen formulas carefully.
        # C00 = M5 - M4 + M6 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M6
        
        # M1 = A00(B00+B11)
        u[0, 0] = 1
        v[0, 0] = 1
        v[3, 0] = 1
        w[0, 0] = 1 # C00
        w[1, 0] = 1 # C11? No, C11 is index 3. w[k*n+i].
        # w indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C00 is w[0]. C11 is w[3].
        # M1 contributes to C00 and C11.
        w[0, 0] = 1
        w[3, 0] = 1
        
        # M2 = (A10+A11)B00
        u[2, 1] = 1 # A10 (index 2)
        u[3, 1] = 1 # A11 (index 3)
        v[0, 1] = 1
        w[2, 1] = 1 # C10
        w[3, 1] = 1 # C11? No.
        # C10 = M2 + M4. C11 = M1 - M3 + M5 + M6.
        # So M2 contributes to C10 only?
        # Wait, standard Strassen:
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M6
        # So M2 -> C10.
        w[2, 1] = 1
        
        # M3 = A00(B01-B11)
        u[0, 2] = 1
        v[1, 2] = 1
        v[3, 2] = -1
        w[1, 2] = 1 # C01
        w[3, 2] = -1 # C11
        
        # M4 = A11(B10-B00)
        u[3, 3] = 1
        v[2, 3] = 1
        v[0, 3] = -1
        w[0, 3] = -1 # C00
        w[2, 3] = 1 # C10
        
        # M5 = (A00+A11)(B00+B11)
        u[0, 4] = 1
        u[3, 4] = 1
        v[0, 4] = 1
        v[3, 4] = 1
        w[0, 4] = 1 # C00
        w[3, 4] = 1 # C11
        
        # M6 = (A01-A11)(B10+B11)
        u[1, 5] = 1
        u[3, 5] = -1
        v[2, 5] = 1
        v[3, 5] = 1
        w[0, 5] = -1 # C00? No.
        # C00 = M5 - M4 + M6 - M7
        # So M6 contributes to C00 with +1?
        # Let's re-verify C00.
        # C00 = M5 - M4 + M6 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M6
        #
        # M6 term in C00 is +1.
        # M6 term in C11 is +1.
        w[0, 5] = 1
        w[3, 5] = 1
        
        # M7 = (A00+A10)(B01+B10)
        u[0, 6] = 1
        u[2, 6] = 1
        v[1, 6] = 1
        v[2, 6] = 1
        w[0, 6] = -1 # C00
        w[1, 6] = 1 # C01
        
        return u, v, w

    # 2. Schoolbook 2x2x3
    def schoolbook_223():
        r = 12
        u = np.zeros((4, r))
        v = np.zeros((6, r))
        w = np.zeros((6, r))
        
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u[i * 2 + j, c] = 1
                    v[j * 3 + k, c] = 1
                    w[k * 2 + i, c] = 1
                    c += 1
        return u, v, w

    # Decomposition 1: A1 B1
    # A1 is 2x2 (i:0-1, j:0-1)
    # B1 is 2x5 (j:0-1, k:0-4)
    # Split B1 into B1a (2x2, k:0-1) and B1b (2x3, k:2-4)
    
    # P1a: A1 B1a (2x2x2)
    u1a, v1a, w1a = strassen_222()
    add_subproblem(2, 2, 2, 0, 0, 0, u1a, v1a, w1a, 0)
    
    # P1b: A1 B1b (2x2x3)
    u1b, v1b, w1b = schoolbook_223()
    add_subproblem(2, 2, 3, 0, 0, 2, u1b, v1b, w1b, 7)
    
    # Decomposition 2: A2 B2
    # A2 is 2x2 (i:0-1, j:2-3)
    # B2 is 2x5 (j:2-3, k:0-4)
    # Split B2 into B2a (2x2, k:0-1) and B2b (2x3, k:2-4)
    
    # P2a: A2 B2a (2x2x2)
    u2a, v2a, w2a = strassen_222()
    add_subproblem(2, 2, 2, 0, 2, 0, u2a, v2a, w2a, 19)
    
    # P2b: A2 B2b (2x2x3)
    u2b, v2b, w2b = schoolbook_223()
    add_subproblem(2, 2, 3, 0, 2, 2, u2b, v2b, w2b, 26)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
