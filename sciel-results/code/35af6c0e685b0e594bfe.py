import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U[i*m+j, r]
    # V[j*p+k, r] -> here p=2 for local
    # W[k*n+i, r] -> here n=2 for local
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_str = np.zeros((4, 7), dtype=int)
    V_str = np.zeros((4, 7), dtype=int) # Local p=2
    W_str = np.zeros((4, 7), dtype=int) # Local n=2
    
    # r=0
    U_str[0, 0] = 1; U_str[3, 0] = 1 # A00, A11
    V_str[0, 0] = 1; V_str[3, 0] = 1 # B00, B11 (j=0,k=0 and j=1,k=1) -> idx 0, 3
    W_str[0, 0] = 1; W_str[1, 0] = 1 # C00, C10? No, W[k*n+i].
    # C00 (k=0,i=0) -> idx 0. C11 (k=1,i=1) -> idx 3.
    # M1 contributes to C00 and C11?
    # C00 = M1 - M2 - M4 + M6
    # C11 = M1 - M3 - M5 + M7
    # Let's check standard Strassen.
    # C00 = M1 - M2 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M7
    
    # So W entries (coefficients for C):
    # r=0 (M1): C00 (+1), C11 (+1)
    W_str[0, 0] = 1
    W_str[3, 0] = 1 # k=1, i=1 -> idx 3
    
    # r=1 (M2): C00 (-1), C10 (+1)
    U_str[2, 1] = 1; U_str[3, 1] = 1 # A10, A11
    V_str[0, 1] = 1 # B00
    W_str[0, 1] = -1
    W_str[2, 1] = 1 # k=1, i=0 -> idx 2
    
    # r=2 (M3): C11 (-1)
    U_str[0, 2] = 1 # A00
    V_str[2, 2] = -1; V_str[3, 2] = -1 # B10, B11 -> j=1,k=0 (idx 2), j=1,k=1 (idx 3)
    # Wait, V index j*p+k. p=2.
    # j=1, k=0 -> idx 2. j=1, k=1 -> idx 3.
    W_str[3, 2] = -1
    
    # r=3 (M4): C00 (-1), C10 (+1)
    U_str[3, 3] = 1 # A11
    V_str[1, 3] = 1; V_str[0, 3] = -1 # B01, B00 -> j=0,k=1 (idx 1), j=0,k=0 (idx 0)
    W_str[0, 3] = -1
    W_str[2, 3] = 1
    
    # r=4 (M5): C01 (+1), C11 (-1)
    U_str[0, 4] = 1; U_str[1, 4] = 1 # A00, A01
    V_str[3, 4] = 1 # B11
    W_str[1, 4] = 1 # C01 (k=0, i=1) -> idx 1
    W_str[3, 4] = -1
    
    # r=5 (M6): C00 (+1), C01 (+1)
    U_str[2, 5] = 1; U_str[0, 5] = -1 # A10, A00
    V_str[0, 5] = 1; V_str[1, 5] = 1 # B00, B01
    W_str[0, 5] = 1
    W_str[1, 5] = 1
    
    # r=6 (M7): C01 (+1), C11 (+1)
    U_str[1, 6] = 1; U_str[3, 6] = -1 # A01, A11
    V_str[2, 6] = 1; V_str[3, 6] = 1 # B10, B11
    W_str[1, 6] = 1
    W_str[3, 6] = 1
    
    # Schoolbook for 2x2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # 4 terms.
    U_sch = np.zeros((4, 4), dtype=int)
    V_sch = np.zeros((2, 4), dtype=int) # Local p=1
    W_sch = np.zeros((2, 4), dtype=int) # Local n=2
    
    # r=0: A00 B0 -> C0
    U_sch[0, 0] = 1
    V_sch[0, 0] = 1 # j=0, k=0
    W_sch[0, 0] = 1 # k=0, i=0
    
    # r=1: A01 B1 -> C0
    U_sch[1, 1] = 1
    V_sch[1, 1] = 1 # j=1, k=0
    W_sch[0, 1] = 1
    
    # r=2: A10 B0 -> C1
    U_sch[2, 2] = 1
    V_sch[0, 2] = 1
    W_sch[1, 2] = 1 # k=0, i=1
    
    # r=3: A11 B1 -> C1
    U_sch[3, 3] = 1
    V_sch[1, 3] = 1
    W_sch[1, 3] = 1

    # Assemble
    rank = 5 * 7 + 4
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((22, rank), dtype=int)
    W = np.zeros((22, rank), dtype=int)
    
    col = 0
    for b in range(5):
        # Strassen block for cols 2b, 2b+1
        U[:, col:col+7] = U_str
        # V mapping
        for r in range(7):
            for j in range(2):
                for k_loc in range(2):
                    v_val = V_str[j*2 + k_loc, r]
                    if v_val != 0:
                        k_glob = 2*b + k_loc
                        v_idx = j * p + k_glob
                        V[v_idx, col + r] = v_val
        # W mapping
        for r in range(7):
            for k_loc in range(2):
                for i in range(2):
                    w_val = W_str[k_loc*2 + i, r]
                    if w_val != 0:
                        k_glob = 2*b + k_loc
                        w_idx = k_glob * n + i
                        W[w_idx, col + r] = w_val
        col += 7
        
    # Schoolbook block for col 10
    U[:, col:col+4] = U_sch
    for r in range(4):
        for j in range(2):
            v_val = V_sch[j, r]
            if v_val != 0:
                k_glob = 10
                v_idx = j * p + k_glob
                V[v_idx, col + r] = v_val
    for r in range(4):
        for i in range(2):
            w_val = W_sch[i, r] # W_sch index is k_loc*n + i. k_loc=0. So idx i.
            if w_val != 0:
                k_glob = 10
                w_idx = k_glob * n + i
                W[w_idx, col + r] = w_val
    col += 4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
