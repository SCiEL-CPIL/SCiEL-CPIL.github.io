import time

DIMENSIONS = [4, 4, 9]

_DEADLINE_FRACTION = 0.90
_WALL_BUDGET_S = 120.0

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-114 decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen-like recursion (rank 49 for 4x4x4) and schoolbook (rank 16 for 4x4x1).
    """
    _t0 = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Define 2x2x2 Winograd (Rank 7)
    # U22, V22, W22 are 4x7
    # Indices for 2x2: i in 0..1, j in 0..1, k in 0..1
    # Flattened A index: i*2 + j
    # Flattened B index: j*2 + k
    # Flattened C index: k*2 + i (Trace convention: W[k*n+i, r] -> for 2x2, W[k*2+i])
    
    U22 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 1, 0, 0, 0, 1, 1],  # i=0, j=1
        [1, 1, 0, 1, 0, -1, -1],# i=1, j=0
        [0, 0, -1, 1, 1, 0, 0]  # i=1, j=1
    ]
    # Wait, let's verify the 2x2x2 Winograd matrices.
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U (A coefficients):
    # M1: a11:1, a22:1
    # M2: a21:1, a22:1
    # M3: a11:1
    # M4: a22:1
    # M5: a11:1, a12:1
    # M6: a21:1, a11:-1
    # M7: a12:1, a22:-1
    
    # Rows: (0,0), (0,1), (1,0), (1,1)
    U22 = [
        [1, 0, 1, 0, 1, -1, 0], # a11 (0,0)
        [0, 0, 0, 0, 1, 0, 1],  # a12 (0,1)
        [0, 1, 0, 0, 0, 1, 0],  # a21 (1,0)
        [1, 1, 0, 1, 0, 0, -1]  # a22 (1,1)
    ]
    
    # V (B coefficients):
    # M1: b11:1, b22:1
    # M2: b11:1
    # M3: b12:1, b22:-1
    # M4: b21:1, b11:-1
    # M5: b22:1
    # M6: b11:1, b12:1
    # M7: b21:1, b22:1
    
    # Rows: (0,0), (0,1), (1,0), (1,1) corresponding to b11, b12, b21, b22
    V22 = [
        [1, 1, 0, -1, 0, 1, 0], # b11 (0,0)
        [0, 0, 1, 0, 0, 1, 0],  # b12 (0,1)
        [0, 0, 0, 1, 0, 0, 1],  # b21 (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # b22 (1,1)
    ]
    
    # W (C coefficients):
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Rows: (0,0), (0,1), (1,0), (1,1) corresponding to c11, c12, c21, c22
    # Convention W[k*n+i]. For 2x2, n=2.
    # c11: k=0, i=0 -> idx 0
    # c12: k=1, i=0 -> idx 2? No, k*n+i.
    # If C is 2x2, C[i,k].
    # Trace convention W[k*n+i].
    # c11 (i=0, k=0) -> 0*2+0 = 0
    # c12 (i=0, k=1) -> 1*2+0 = 2
    # c21 (i=1, k=0) -> 0*2+1 = 1
    # c22 (i=1, k=1) -> 1*2+1 = 3
    
    W22 = [
        [1, 0, 0, 1, -1, 0, 1], # c11 (0)
        [0, 1, 0, 1, 0, 0, 0],  # c21 (1)
        [0, 0, 1, 0, 1, 0, 0],  # c12 (2)
        [1, -1, 1, 0, 0, 1, 0]  # c22 (3)
    ]
    
    # 2. Construct 4x4x4 Strassen (Rank 49)
    # Strassen coefficients for A (L_s)
    # L1 = A11 + A22
    # L2 = A21 + A22
    # L3 = A11
    # L4 = A22
    # L5 = A11 + A12
    # L6 = A21 - A11
    # L7 = A12 - A22
    
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # Indices i1, j1
    Strassen_A = [
        [[1, 0], [0, 1]],  # L1: A00, A11
        [[0, 0], [1, 1]],  # L2: A10, A11
        [[1, 0], [0, 0]],  # L3: A00
        [[0, 0], [0, 1]],  # L4: A11
        [[1, 1], [0, 0]],  # L5: A00, A01
        [[-1, 0], [1, 0]], # L6: A00(-1), A10
        [[0, 1], [0, -1]]  # L7: A01, A11(-1)
    ]
    
    Strassen_B = [
        [[1, 0], [0, 1]],  # R1: B00, B11
        [[1, 0], [0, 0]],  # R2: B00
        [[0, 1], [0, -1]], # R3: B01, B11(-1)
        [[-1, 0], [1, 0]], # R4: B00(-1), B10
        [[0, 0], [0, 1]],  # R5: B11
        [[1, 1], [0, 0]],  # R6: B00, B01
        [[0, 1], [0, 1]]   # R7: B01, B11
    ]
    
    Strassen_C = [
        [1, 0, 0, 1],  # C00: M1, M4, M7(-1)? No.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Indices: C00, C01, C10, C11
        [1, 0, 0, 1],  # C00 (M1, M4, -M5, M7) -> 1, 0, 0, 1?
        # Let's list M1..M7 coefficients for each C block
        # C00: M1:1, M4:1, M5:-1, M7:1
        # C01: M3:1, M5:1
        # C10: M2:1, M4:1
        # C11: M1:1, M2:-1, M3:1, M6:1
        [1, 0, 0, 1],  # C00
        [0, 0, 1, 0],  # C01
        [0, 1, 0, 1],  # C10
        [1, -1, 1, 0]  # C11
    ]
    # Note: Strassen_C[s][c_idx] is the coefficient of M_s in C_block.
    # My list above was C00, C01, C10, C11.
    # Let's re-verify C00: M1+M4-M5+M7.
    # M1:1, M2:0, M3:0, M4:1, M5:-1, M6:0, M7:1.
    # My array [1, 0, 0, 1] only has 4 elements. I need 7.
    
    Strassen_C = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    U44 = []
    V44 = []
    W44 = []
    
    for s in range(7):
        for t in range(7):
            u_col = [0] * 16
            v_col = [0] * 16
            w_col = [0] * 16
            
            for i1 in range(2):
                for j1 in range(2):
                    coeff_a = Strassen_A[s][i1][j1]
                    if coeff_a != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2) # A is 4x4, row i, col j
                                # Wait, U rows are n*m = 16. Index i*m+j.
                                # i = i1*2 + i2, j = j1*2 + j2.
                                # u_idx = i * 4 + j.
                                u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                                u_col[u_idx] += coeff_a * U22[i2 * 2 + j2][t]
                                
            for j1 in range(2):
                for k1 in range(2):
                    coeff_b = Strassen_B[s][j1][k1]
                    if coeff_b != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2) # B is 4x4, row j, col k
                                # j = j1*2 + j2, k = k1*2 + k2.
                                # v_idx = j * 4 + k.
                                v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                                v_col[v_idx] += coeff_b * V22[j2 * 2 + k2][t]
                                
            for i1 in range(2):
                for k1 in range(2):
                    coeff_c = Strassen_C[s][i1 * 2 + k1] # C block index i1*2+k1? No.
                    # C blocks are C00, C01, C10, C11.
                    # Index in Strassen_C is i1*2 + k1.
                    if coeff_c != 0:
                        for i2 in range(2):
                            for k2 in range(2):
                                w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2) # W is p*n = 16. Index k*n+i.
                                # k = k1*2 + k2, i = i1*2 + i2.
                                # w_idx = k * 4 + i.
                                w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                                w_col[w_idx] += coeff_c * W22[k2 * 2 + i2][t]
                                
            U44.append(u_col)
            V44.append(v_col)
            W44.append(w_col)
            
    # 3. Construct 4x4x1 Schoolbook (Rank 16)
    U41 = []
    V41 = []
    W41 = []
    for i in range(4):
        for j in range(4):
            u_col = [0] * 16
            v_col = [0] * 4 # Only k=8, so j*4+8. But V for this part is just 4 rows?
            # In the final assembly, V will have 36 rows.
            # Here we just define the local V for j in 0..3, k=8.
            # Local V index: j.
            w_col = [0] * 4 # k=8, i in 0..3. Local W index: i.
            
            u_col[i * 4 + j] = 1
            v_col[j] = 1
            w_col[i] = 1
            
            U41.append(u_col)
            V41.append(v_col)
            W41.append(w_col)
            
    # 4. Assemble 4x4x9
    rank = 49 + 49 + 16
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    col = 0
    # Part 1: k=0..3
    for c in range(49):
        for i in range(16):
            U[i][col] = U44[c][i]
        for j in range(4):
            for k in range(4):
                V[j * 4 + k][col] = V44[c][j * 4 + k]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][col] = W44[c][k * 4 + i]
        col += 1
        
    # Part 2: k=4..7
    for c in range(49):
        for i in range(16):
            U[i][col] = U44[c][i]
        for j in range(4):
            for k in range(4, 8):
                # Map local k (0..3) to global k (4..7)
                V[j * 4 + k][col] = V44[c][j * 4 + (k - 4)]
        for k in range(4, 8):
            for i in range(4):
                W[k * 4 + i][col] = W44[c][(k - 4) * 4 + i]
        col += 1
        
    # Part 3: k=8
    for c in range(16):
        for i in range(16):
            U[i][col] = U41[c][i]
        for j in range(4):
            V[j * 4 + 8][col] = V41[c][j]
        for i in range(4):
            W[8 * 4 + i][col] = W41[c][i]
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
