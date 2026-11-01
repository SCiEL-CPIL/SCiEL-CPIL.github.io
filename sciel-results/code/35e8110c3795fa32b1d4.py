import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    Total rank = 49.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2): A[00], A[01], A[10], A[11]
    # U_forms[s][i] where i=0:A00, 1:A01, 2:A10, 3:A11
    U_forms = [
        [1, 0, 0, 1],   # M1 = A00 + A11
        [0, 0, 1, 1],   # M2 = A10 + A11
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = A00 + A01
        [-1, 0, 1, 0],  # M6 = A10 - A00
        [0, 1, 0, -1]   # M7 = A01 - A11
    ]
    
    # For each block r (columns 2r, 2r+1 of B):
    # B local: e=B[0,2r], f=B[0,2r+1], g=B[1,2r], h=B[1,2r+1]
    # C local: C[0,2r]=C00, C[0,2r+1]=C01, C[1,2r]=C10, C[1,2r+1]=C11
    #
    # Strassen:
    # M1 = (A00+A11)(e+h)  -> C00 += M1, C11 += M1
    # M2 = (A10+A11)e      -> C10 += M2, C11 -= M2
    # M3 = A00(f-h)        -> C01 += M3, C11 += M3
    # M4 = A11(g-e)        -> C00 += M4, C10 += M4
    # M5 = (A00+A01)h      -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(e+f)  -> C11 += M6
    # M7 = (A01-A11)(g+h)  -> C00 += M7
    
    # V forms per Strassen term (local B indices: 0:e, 1:f, 2:g, 3:h)
    V_forms = [
        {0: 1, 3: 1},   # M1: e+h
        {0: 1},         # M2: e
        {1: 1, 3: -1},  # M3: f-h
        {2: 1, 0: -1},  # M4: g-e
        {3: 1},         # M5: h
        {0: 1, 1: 1},   # M6: e+f
        {2: 1, 3: 1}    # M7: g+h
    ]
    
    # W forms per Strassen term (local C indices: 0:C00, 1:C10, 2:C01, 3:C11)
    W_forms = [
        {0: 1, 3: 1},   # M1: C00 + C11
        {1: 1, 3: -1},  # M2: C10 - C11
        {2: 1, 3: 1},   # M3: C01 + C11
        {0: 1, 1: 1},   # M4: C00 + C10
        {0: -1, 2: 1},  # M5: -C00 + C01
        {3: 1},         # M6: C11
        {0: 1}          # M7: C00
    ]
    
    for r in range(num_blocks):
        for s in range(7):
            col = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][col] = U_forms[s][i]
            
            # Fill V
            for v_local, coeff in V_forms[s].items():
                # v_local: 0->B[0,2r], 1->B[0,2r+1], 2->B[1,2r], 3->B[1,2r+1]
                row = v_local // 2
                col_local = v_local % 2
                v_global = row * p + (2 * r + col_local)
                V[v_global][col] = coeff
            
            # Fill W
            for w_local, coeff in W_forms[s].items():
                # w_local: 0->C[0,2r], 1->C[1,2r], 2->C[0,2r+1], 3->C[1,2r+1]
                # Global W index: k*n + i where C[i,k]
                if w_local == 0:
                    w_global = (2 * r) * n + 0
                elif w_local == 1:
                    w_global = (2 * r) * n + 1
                elif w_local == 2:
                    w_global = (2 * r + 1) * n + 0
                elif w_local == 3:
                    w_global = (2 * r + 1) * n + 1
                W[w_global][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
