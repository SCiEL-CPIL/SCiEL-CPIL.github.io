import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen factors for 2x2x2
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7
    
    # M1 = A00 * (B11 - B01)
    # M2 = A10 * (B01 + B11)
    # M3 = (A00 + A01) * B01
    # M4 = (A10 + A11) * B10
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 - A10) * (B00 + B01)
    
    # Indices for U (A): i*2 + j
    # Indices for V (B): j*2 + k
    # Indices for W (C): k*2 + i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0 (M1)
    U_s[0, 0] = 1  # A00
    V_s[3, 0] = 1  # B11 (j=1, k=1 -> 1*2+1=3)
    V_s[1, 0] = -1 # B01 (j=0, k=1 -> 0*2+1=1)
    W_s[2, 0] = 1  # C01 (k=1, i=0 -> 1*2+0=2)
    W_s[3, 0] = 1  # C11 (k=1, i=1 -> 1*2+1=3)
    
    # r=1 (M2)
    U_s[2, 1] = 1  # A10 (i=1, j=0 -> 1*2+0=2)
    V_s[1, 1] = 1  # B01
    V_s[3, 1] = 1  # B11
    W_s[2, 1] = 1  # C01
    W_s[3, 1] = -1 # C11? No.
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M7
    # So M2 contributes +1 to C01, 0 to others?
    # Wait, C01 = M1 + M2. So W for M2 at C01 is 1.
    # Does M2 contribute to C11? No.
    # So W_s[2, 1] = 1, others 0.
    
    # Let's re-verify C formulas.
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # Correct W_s:
    W_s = np.zeros((4, 7))
    # M1 (r=0)
    W_s[2, 0] = 1 # C01
    W_s[3, 0] = 1 # C11
    # M2 (r=1)
    W_s[2, 1] = 1 # C01
    # M3 (r=2)
    W_s[0, 2] = 1 # C10 (k=0, i=1 -> 0*2+1=1? No. k=0, i=1 -> 1. C10 is i=1, k=0. Index 1*2+0=2? No.
    # W index is k*n + i. n=2.
    # C10: i=1, k=0. Index 0*2 + 1 = 1.
    # C11: i=1, k=1. Index 1*2 + 1 = 3.
    # C01: i=0, k=1. Index 1*2 + 0 = 2.
    # C00: i=0, k=0. Index 0*2 + 0 = 0.
    
    # Let's map W indices carefully.
    # C00 (i=0, k=0) -> 0
    # C01 (i=0, k=1) -> 2
    # C10 (i=1, k=0) -> 1
    # C11 (i=1, k=1) -> 3
    
    W_s[2, 0] = 1 # C01
    W_s[3, 0] = 1 # C11
    W_s[2, 1] = 1 # C01
    W_s[1, 2] = 1 # C10
    W_s[3, 2] = -1 # C11
    W_s[0, 3] = 1 # C00
    W_s[1, 3] = 1 # C10
    W_s[0, 4] = 1 # C00
    W_s[3, 4] = 1 # C11
    W_s[0, 5] = 1 # C00
    W_s[1, 6] = 1 # C11
    
    # U_s
    U_s[0, 0] = 1 # A00
    U_s[2, 1] = 1 # A10
    U_s[0, 2] = 1 # A00
    U_s[1, 2] = 1 # A01
    U_s[2, 3] = 1 # A10
    U_s[3, 3] = 1 # A11
    U_s[0, 4] = 1 # A00
    U_s[3, 4] = 1 # A11
    U_s[1, 5] = 1 # A01
    U_s[3, 5] = -1 # A11
    U_s[0, 6] = 1 # A00
    U_s[2, 6] = -1 # A10
    
    # V_s
    V_s[3, 0] = 1 # B11
    V_s[1, 0] = -1 # B01
    V_s[1, 1] = 1 # B01
    V_s[3, 1] = 1 # B11
    V_s[1, 2] = 1 # B01
    V_s[2, 3] = 1 # B10
    V_s[0, 4] = 1 # B00
    V_s[3, 4] = 1 # B11
    V_s[2, 5] = 1 # B10
    V_s[3, 5] = 1 # B11
    V_s[0, 6] = 1 # B00
    V_s[1, 6] = 1 # B01

    # Fill global matrices
    for s in range(8):
        col_offset = s * 7
        U[:, col_offset:col_offset+7] = U_s
        for j in range(2):
            for k_local in range(2):
                v_idx_local = j * 2 + k_local
                v_idx_global = j * 16 + (s * 2 + k_local)
                V[v_idx_global, col_offset:col_offset+7] = V_s[v_idx_local, :]
        for k_local in range(2):
            for i in range(2):
                w_idx_local = k_local * 2 + i
                w_idx_global = (s * 2 + k_local) * 2 + i
                W[w_idx_global, col_offset:col_offset+7] = W_s[w_idx_local, :]

    return {"U": U, "V": V, "W": W}
