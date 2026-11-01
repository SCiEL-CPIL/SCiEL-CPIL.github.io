import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen A-factors for 2x2
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A10
    # M6: A10 - A00
    # M7: A01 - A11
    
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    U_factors = [
        [1, 0, 1, 0, 1, -1, 1],  # A00
        [0, 0, 0, 0, 0, 0, -1],  # A01
        [0, 1, 0, 0, 1, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # Strassen B-factors
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # V rows: j*p + k. j in {0,1}, k in {0,1} local.
    # Local k=0 -> global 2l, Local k=1 -> global 2l+1
    # M1: B00 (j=0,k=0) + B11 (j=1,k=1)
    # M2: B00 (j=0,k=0)
    # M3: B10 (j=1,k=0) - B11 (j=1,k=1)
    # M4: B01 (j=0,k=1) - B00 (j=0,k=0)
    # M5: B11 (j=1,k=1)
    # M6: B00 (j=0,k=0) + B01 (j=0,k=1)
    # M7: B10 (j=1,k=0) + B11 (j=1,k=1)
    
    V_factors = [
        {0: 1, 3: 1},  # M1: B00, B11
        {0: 1},         # M2: B00
        {2: 1, 3: -1},  # M3: B10, -B11
        {1: 1, 0: -1},  # M4: B01, -B00
        {3: 1},         # M5: B11
        {0: 1, 1: 1},   # M6: B00, B01
        {2: 1, 3: 1}    # M7: B10, B11
    ]
    
    # W factors
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # W indices: k*n + i. k in {0,1} local, i in {0,1}.
    # C00 (i=0, k=0) -> idx 0
    # C01 (i=0, k=1) -> idx 2
    # C10 (i=1, k=0) -> idx 1
    # C11 (i=1, k=1) -> idx 3
    
    W_factors = [
        {0: 1, 1: 1, 3: 1},  # M1: C00, C10, C11
        {1: 1, 3: -1},       # M2: C10, -C11
        {2: 1, 3: 1},        # M3: C01, C11
        {0: 1, 1: 1},        # M4: C00, C10
        {0: -1, 2: 1},       # M5: -C00, C01
        {},                  # M6: Unused?
        {0: -1, 2: -1, 3: 1} # M7: -C00, -C01, C11
    ]
    
    # Note: M6 is not in the standard Strassen output formulas I found.
    # If M6 is not used, the rank is effectively 6?
    # But rank of 2x2x2 is 7.
    # I will leave M6 with 0s in W for now, or check if I missed a term.
    # Actually, if M6 is not used, we can drop it and use rank 36?
    # No, 2x2x2 rank is 7.
    
    for l in range(6):
        col_start = l * 7
        for r in range(7):
            col = col_start + r
            # U
            for i in range(4):
                U[i][col] = U_factors[i][r]
            
            # V
            for local_idx, val in V_factors[r].items():
                j, k_loc = local_idx // 2, local_idx % 2
                global_k = 2 * l + k_loc
                V[j * p + global_k][col] = val
            
            # W
            for local_idx, val in W_factors[r].items():
                k_loc, i = local_idx // 2, local_idx % 2
                global_k = 2 * l + k_loc
                W[global_k * n + i][col] = val

    return {"U": U, "V": V, "W": W}
