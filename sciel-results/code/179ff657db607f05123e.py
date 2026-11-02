import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85  # 49 (Strassen 4x4) + 16 (Tail 1) + 20 (Tail 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar product term
    # a_coeffs: list of (global_A_idx, coeff)
    # b_coeffs: list of (global_B_idx, coeff)
    # c_idx: global_C_idx
    # c_coeff: coeff
    def add_term(a_coeffs, b_coeffs, c_idx, c_coeff):
        nonlocal col
        for idx, c in a_coeffs:
            U[idx][col] += c
        for idx, c in b_coeffs:
            V[idx][col] += c
        W[c_idx][col] += c_coeff
        col += 1

    # Helper to get global A index for local 4x4 block (rows 0-3, cols 0-3)
    def a_idx(r, c):
        return r * 5 + c
        
    # Helper to get global B index for local 4x4 block (rows 0-3, cols 0-3)
    def b_idx(r, c):
        return r * 5 + c
        
    # Helper to get global C index for local 4x4 block (rows 0-3, cols 0-3)
    # C is 4x5. Core is C[0:4, 0:4].
    # W[k*n + i, r] -> W[col_idx, row_idx] in C?
    # Task: W[k*n+i, r]. C[i, k] = sum U[i,j] V[j,k] W[k,i]?
    # Standard: C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor: T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}?
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # C[i, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # So for a term A[a_idx] * B[b_idx] contributing to C[c_idx]:
    # U[a_idx, col] = 1
    # V[b_idx, col] = 1
    # W[c_idx, col] = 1
    # where c_idx = k*n + i.
    
    def c_idx(i, k):
        return k * 4 + i

    # Strassen 2x2 decomposition helper
    # L, R are 2x2 lists of (coeff, local_a_idx) or similar?
    # Let's represent L and R as 2x2 arrays of linear forms.
    # Linear form: list of (global_idx, coeff)
    
    def strassen_2x2(L, R, C_target, C_target_coeffs):
        # L, R are 2x2 arrays of linear forms (lists of (idx, coeff))
        # C_target is a 2x2 array of global C indices
        # C_target_coeffs is a 2x2 array of coefficients for the contribution to C
        
        # m1 = (L00 + L11)(R00 + R11)
        l1 = []
        for idx, c in L[0][0]: l1.append((idx, c))
        for idx, c in L[1][1]: l1.append((idx, c))
        r1 = []
        for idx, c in R[0][0]: r1.append((idx, c))
        for idx, c in R[1][1]: r1.append((idx, c))
        
        # m2 = (L10 + L11)R00
        l2 = []
        for idx, c in L[1][0]: l2.append((idx, c))
        for idx, c in L[1][1]: l2.append((idx, c))
        r2 = R[0][0]
        
        # m3 = L00(R01 - R11)
        l3 = L[0][0]
        r3 = []
        for idx, c in R[0][1]: r3.append((idx, c))
        for idx, c in R[1][1]: r3.append((idx, -c))
        
        # m4 = L11(R10 - R00)
        l4 = L[1][1]
        r4 = []
        for idx, c in R[1][0]: r4.append((idx, c))
        for idx, c in R[0][0]: r4.append((idx, -c))
        
        # m5 = (L00 + L01)R11
        l5 = []
        for idx, c in L[0][0]: l5.append((idx, c))
        for idx, c in L[0][1]: l5.append((idx, c))
        r5 = R[1][1]
        
        # m6 = (L10 - L00)(R00 + R01)
        l6 = []
        for idx, c in L[1][0]: l6.append((idx, c))
        for idx, c in L[0][0]: l6.append((idx, -c))
        r6 = []
        for idx, c in R[0][0]: r6.append((idx, c))
        for idx, c in R[0][1]: r6.append((idx, c))
        
        # m7 = (L01 - L11)(R10 + R11)
        l7 = []
        for idx, c in L[0][1]: l7.append((idx, c))
        for idx, c in L[1][1]: l7.append((idx, -c))
        r7 = []
        for idx, c in R[1][0]: r7.append((idx, c))
        for idx, c in R[1][1]: r7.append((idx, c))
        
        # Contributions to C
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        add_term(l1, r1, C_target[0][0], C_target_coeffs[0][0])
        add_term(l4, r4, C_target[0][0], C_target_coeffs[0][0])
        add_term(l5, r5, C_target[0][0], -C_target_coeffs[0][0])
        add_term(l7, r7, C_target[0][0], C_target_coeffs[0][0])
        
        add_term(l3, r3, C_target[0][1], C_target_coeffs[0][1])
        add_term(l5, r5, C_target[0][1], C_target_coeffs[0][1])
        
        add_term(l2, r2, C_target[1][0], C_target_coeffs[1][0])
        add_term(l4, r4, C_target[1][0], C_target_coeffs[1][0])
        
        add_term(l1, r1, C_target[1][1], C_target_coeffs[1][1])
        add_term(l2, r2, C_target[1][1], -C_target_coeffs[1][1])
        add_term(l3, r3, C_target[1][1], C_target_coeffs[1][1])
        add_term(l6, r6, C_target[1][1], C_target_coeffs[1][1])

    # Define the 7 intermediate 2x2 products for 4x4 Strassen
    # A blocks: A11, A12, A21, A22
    # B blocks: B11, B12, B21, B22
    
    # A11: (0,0), (0,1), (1,0), (1,1) in local 4x4
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    def get_block_A(r0, c0):
        # Returns 2x2 array of linear forms
        block = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(a_idx(r0+i, c0+j), 1)])
            block.append(row)
        return block

    def get_block_B(r0, c0):
        block = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append([(b_idx(r0+i, c0+j), 1)])
            block.append(row)
        return block

    # P1 = (A11 + A22)(B11 + B22)
    L1 = [[None for _ in range(2)] for _ in range(2)]
    R1 = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            L1[i][j] = get_block_A(0,0)[i][j] + get_block_A(2,2)[i][j]
            R1[i][j] = get_block_B(0,0)[i][j] + get_block_B(2,2)[i][j]
            
    # P2 = (A21 + A22)B11
    L2 = [[None for _ in range(2)] for _ in range(2)]
    R2 = get_block_B(0,0)
    for i in range(2):
        for j in range(2):
            L2[i][j] = get_block_A(2,0)[i][j] + get_block_A(2,2)[i][j]
            
    # P3 = A11(B12 - B22)
    L3 = get_block_A(0,0)
    R3 = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            R3[i][j] = get_block_B(0,2)[i][j] + [(idx, -c) for idx, c in get_block_B(2,2)[i][j]]
            
    # P4 = A22(B21 - B11)
    L4 = get_block_A(2,2)
    R4 = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            R4[i][j] = get_block_B(2,0)[i][j] + [(idx, -c) for idx, c in get_block_B(0,0)[i][j]]
            
    # P5 = (A11 + A12)B22
    L5 = [[None for _ in range(2)] for _ in range(2)]
    R5 = get_block_B(2,2)
    for i in range(2):
        for j in range(2):
            L5[i][j] = get_block_A(0,0)[i][j] + get_block_A(0,2)[i][j]
            
    # P6 = (A21 - A11)(B11 + B12)
    L6 = [[None for _ in range(2)] for _ in range(2)]
    R6 = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            L6[i][j] = get_block_A(2,0)[i][j] + [(idx, -c) for idx, c in get_block_A(0,0)[i][j]]
            R6[i][j] = get_block_B(0,0)[i][j] + get_block_B(0,2)[i][j]
            
    # P7 = (A12 - A22)(B21 + B22)
    L7 = [[None for _ in range(2)] for _ in range(2)]
    R7 = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            L7[i][j] = get_block_A(0,2)[i][j] + [(idx, -c) for idx, c in get_block_A(2,2)[i][j]]
            R7[i][j] = get_block_B(2,0)[i][j] + get_block_B(2,2)[i][j]

    # C target indices for the 4x4 core
    # C is 4x4.
    # P1 contributes to C00, C11
    # P2 contributes to C10, C11
    # P3 contributes to C01, C11
    # P4 contributes to C00, C10
    # P5 contributes to C00, C01
    # P6 contributes to C11
    # P7 contributes to C00
    
    # Strassen 4x4 Reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map the 2x2 result of each Pk to the 4x4 C.
    # P1 is 2x2. Its (0,0) goes to C(0,0). (0,1) to C(0,1)?
    # No, P1 is the product of two 2x2 blocks.
    # (A11+A22) is 2x2. (B11+B22) is 2x2.
    # The result is a 2x2 matrix.
    # This 2x2 matrix corresponds to the top-left 2x2 of the 4x4 C?
    # No.
    # A11 is A[0:2, 0:2]. A22 is A[2:4, 2:4].
    # B11 is B[0:2, 0:2]. B22 is B[2:4, 2:4].
    # (A11+A22)(B11+B22) is not a block of C.
    # It is a linear combination.
    # The Strassen algorithm for 4x4 works by treating the 4x4 matrix as 2x2 blocks of 2x2 matrices.
    # The multiplication of these blocks is done using Strassen.
    # The result of each Strassen product Mk is a 2x2 matrix.
    # These 2x2 matrices are then combined to form the 4x4 C.
    # Specifically:
    # C[0:2, 0:2] = M1[0:2, 0:2] + M4[0:2, 0:2] - M5[0:2, 0:2] + M7[0:2, 0:2]
    # C[0:2, 2:4] = M3[0:2, 0:2] + M5[0:2, 0:2]
    # C[2:4, 0:2] = M2[0:2, 0:2] + M4[0:2, 0:2]
    # C[2:4, 2:4] = M1[0:2, 0:2] - M2[0:2, 0:2] + M3[0:2, 0:2] + M6[0:2, 0:2]
    
    # So, for each Mk, its 2x2 result contributes to specific 2x2 blocks of C.
    
    # Define C target indices for a 2x2 block at (r0, c0) in 4x4 C
    def get_c_block_indices(r0, c0):
        return [[c_idx(r0+i, c0+j) for j in range(2)] for i in range(2)]
        
    # C blocks:
    # C_TL: (0,0)
    # C_TR: (0,2)
    # C_BL: (2,0)
    # C_BR: (2,2)
    
    C_TL = get_c_block_indices(0, 0)
    C_TR = get_c_block_indices(0, 2)
    C_BL = get_c_block_indices(2, 0)
    C_BR = get_c_block_indices(2, 2)
    
    # Coefficients are all 1 or -1.
    # P1 -> C_TL (1), C_BR (1)
    strassen_2x2(L1, R1, C_TL, [[1,1],[1,1]])
    strassen_2x2(L1, R1, C_BR, [[1,1],[1,1]])
    
    # P2 -> C_BL (1), C_BR (-1)
    strassen_2x2(L2, R2, C_BL, [[1,1],[1,1]])
    strassen_2x2(L2, R2, C_BR, [[-1,-1],[-1,-1]])
    
    # P3 -> C_TR (1), C_BR (1)
    strassen_2x2(L3, R3, C_TR, [[1,1],[1,1]])
    strassen_2x2(L3, R3, C_BR, [[1,1],[1,1]])
    
    # P4 -> C_TL (1), C_BL (1)
    strassen_2x2(L4, R4, C_TL, [[1,1],[1,1]])
    strassen_2x2(L4, R4, C_BL, [[1,1],[1,1]])
    
    # P5 -> C_TL (-1), C_TR (1)
    strassen_2x2(L5, R5, C_TL, [[-1,-1],[-1,-1]])
    strassen_2x2(L5, R5, C_TR, [[1,1],[1,1]])
    
    # P6 -> C_BR (1)
    strassen_2x2(L6, R6, C_BR, [[1,1],[1,1]])
    
    # P7 -> C_TL (1)
    strassen_2x2(L7, R7, C_TL, [[1,1],[1,1]])
    
    # Tail 1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            a_coeffs = [(a_idx(i, j), 1)]
            b_coeffs = [(b_idx(j, 4), 1)]
            c_idx_val = c_idx(i, 4)
            add_term(a_coeffs, b_coeffs, c_idx_val, 1)
            
    # Tail 2: A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # C[i, k] = A[i, 4] B[4, k]
    for i in range(4):
        for k in range(5):
            a_coeffs = [(a_idx(i, 4), 1)]
            b_coeffs = [(b_idx(4, k), 1)]
            c_idx_val = c_idx(i, k)
            add_term(a_coeffs, b_coeffs, c_idx_val, 1)

    return {"U": U, "V": V, "W": W}
