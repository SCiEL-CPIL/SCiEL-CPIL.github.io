"""
Matrix multiplication 4x4 by 4x5 with rank-65 decomposition.
Uses Strassen's algorithm for the 4x4x4 part (rank 49) plus 16 scalar terms for the 5th column.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Uses rank-65 decomposition: Strassen-49 for 4x4x4 + 16 scalar terms for column 5.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 65
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Strassen decomposition for 4x4x4
    # Partition 4x4 matrix into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen for 2x2x2 (7 multiplications):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 4x4x4, we apply this at two levels.
    # Level 1: 4x4 blocks (2x2 of 2x2 blocks)
    # Level 2: 2x2 blocks (2x2 of 1x1 blocks)
    
    # We'll build the decomposition by tracking which scalar (i,j,k) each Strassen term contributes to.
    # Each Strassen term at the top level is a 2x2x2 multiplication of 2x2 blocks.
    # Each 2x2x2 multiplication uses 7 scalar-level Strassen terms.
    # Total: 7 * 7 = 49 terms for 4x4x4.
    
    # Let me define the Strassen-7 decomposition for 2x2x2 first.
    # For a 2x2x2 tensor, the indices are (i,j) for U (4 entries), (j,k) for V (4 entries), (k,i) for W (4 entries).
    # i,j,k in {0,1}
    
    # Strassen-7 terms for 2x2x2:
    # Each term is a linear combination of the input entries.
    # U factors (for A matrix, 2x2):
    #   Term 0: A11 + A22 => indices (0,0) and (1,1)
    #   Term 1: A21 + A22 => indices (1,0) and (1,1)
    #   Term 2: A11 => index (0,0)
    #   Term 3: A22 => index (1,1)
    #   Term 4: A11 + A12 => indices (0,0) and (0,1)
    #   Term 5: A21 - A11 => indices (1,0) and (0,0)
    #   Term 6: A12 - A22 => indices (0,1) and (1,1)
    #
    # V factors (for B matrix, 2x2):
    #   Term 0: B11 + B22 => indices (0,0) and (1,1)
    #   Term 1: B11 => index (0,0)
    #   Term 2: B12 - B22 => indices (0,1) and (1,1)
    #   Term 3: B21 - B11 => indices (1,0) and (0,0)
    #   Term 4: B22 => index (1,1)
    #   Term 5: B11 + B12 => indices (0,0) and (0,1)
    #   Term 6: B21 + B22 => indices (1,0) and (1,1)
    #
    # W factors (for C matrix, 2x2):
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    #
    #   Term 0 (M1): C11 and C22 => W indices (0,0) and (1,1)
    #   Term 1 (M2): C21 and C22 => W indices (1,0) and (1,1)
    #   Term 2 (M3): C12 and C22 => W indices (0,1) and (1,1)
    #   Term 3 (M4): C11 and C21 => W indices (0,0) and (1,0)
    #   Term 4 (M5): C11 and C12 => W indices (0,0) and (0,1)
    #   Term 5 (M6): C22 => W index (1,1)
    #   Term 6 (M7): C11 => W index (0,0)
    
    # Define Strassen-7 factors for 2x2x2
    # U_strassen: 4x7 matrix (rows are i*2+j, cols are term 0-6)
    U_strassen = [
        # i=0,j=0 (A11): terms 0(+1), 2(+1), 4(+1), 5(-1)
        [1, 0, 1, 0, 1, -1, 0],
        # i=0,j=1 (A12): terms 4(+1), 6(+1)
        [0, 0, 0, 0, 1, 0, 1],
        # i=1,j=0 (A21): terms 1(+1), 5(+1)
        [0, 1, 0, 0, 0, 1, 0],
        # i=1,j=1 (A22): terms 0(+1), 1(+1), 3(+1), 6(-1)
        [1, 1, 0, 1, 0, 0, -1],
    ]
    
    # V_strassen: 4x7 matrix (rows are j*2+k, cols are term 0-6)
    V_strassen = [
        # j=0,k=0 (B11): terms 0(+1), 1(+1), 3(-1), 5(+1)
        [1, 1, 0, -1, 0, 1, 0],
        # j=0,k=1 (B12): terms 2(+1), 5(+1)
        [0, 0, 1, 0, 0, 1, 0],
        # j=1,k=0 (B21): terms 3(+1), 6(+1)
        [0, 0, 0, 1, 0, 0, 1],
        # j=1,k=1 (B22): terms 0(+1), 2(-1), 4(+1), 6(+1)
        [1, 0, -1, 0, 1, 0, 1],
    ]
    
    # W_strassen: 4x7 matrix (rows are k*2+i, cols are term 0-6)
    W_strassen = [
        # k=0,i=0 (C11): terms 0(+1), 3(+1), 4(-1), 6(+1)
        [1, 0, 0, 1, -1, 0, 1],
        # k=0,i=1 (C21): terms 1(+1), 3(+1)
        [0, 1, 0, 1, 0, 0, 0],
        # k=1,i=0 (C12): terms 2(+1), 4(+1)
        [0, 0, 1, 0, 1, 0, 0],
        # k=1,i=1 (C22): terms 0(+1), 1(-1), 2(+1), 5(+1)
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Now for 4x4x4, we partition into 2x2 blocks of 2x2 matrices.
    # The 4x4 matrix A has blocks:
    # A = [[A00, A01], [A10, A11]] where each Axx is 2x2
    # Block (I,J) contains elements A[2*I+i][2*J+j] for i,j in {0,1}
    #
    # Similarly for B and C.
    #
    # The Strassen-49 decomposition applies Strassen-7 at the block level,
    # where each "scalar" multiplication of 2x2 blocks is itself a Strassen-7.
    
    # Top-level Strassen-7 operates on 2x2 blocks of 2x2 matrices.
    # For each of the 7 top-level terms, we have a 2x2x2 Strassen-7 decomposition.
    # Total: 7 * 7 = 49 terms.
    
    # Let me define the top-level Strassen factors for the block structure.
    # Top-level U: 4x7 (rows are block indices I*2+J, cols are top-level term 0-6)
    # This is the same as U_strassen but for block indices.
    top_U = U_strassen  # 4x7
    top_V = V_strassen  # 4x7
    top_W = W_strassen  # 4x7
    
    # For each top-level term t (0-6), and each bottom-level term s (0-6),
    # we have a combined term that contributes to the final decomposition.
    
    # The combined U factor for term (t,s):
    #   top_U[I,J,t] * U_strassen[i,j,s]
    # where I = i//2, J = j//2, i = 2*I + i_local, j = 2*J + j_local
    
    # Let me build the 49-term decomposition for 4x4x4.
    
    # For each top-level term t (0-6) and bottom-level term s (0-6):
    #   U_combined[2*I+i_local, 2*J+j_local] += top_U[I*2+J, t] * U_strassen[i_local*2+j_local, s]
    #   V_combined[2*J+j_local, 2*K+k_local] += top_V[J*2+K, t] * V_strassen[j_local*2+k_local, s]
    #   W_combined[2*K+k_local, 2*I+i_local] += top_W[K*2+I, t] * W_strassen[k_local*2+i_local, s]
    
    # Wait, I need to be careful with the indexing.
    # A is 4x4: A[row][col] where row = i, col = j, i,j in {0,1,2,3}
    # Block (I,J) of A contains A[2*I + i_local][2*J + j_local]
    #
    # U matrix has rows indexed by (i,j) = (row, col) of A, so row index = i*4+j
    # V matrix has rows indexed by (j,k) = (col of A, row of B), so row index = j*5+k... 
    # Wait, for 4x4x4, B is 4x4, so V has rows indexed by (j,k) where j is col of A / row of B, k is col of B.
    # V row index = j*4+k
    # W matrix has rows indexed by (k,i) = (col of B, row of C), so row index = k*4+i
    
    # Let me rebuild this more carefully.
    
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # U: 16x49 (rows: i*4+j for i,j in 0..3)
    # V: 16x49 (rows: j*4+k for j,k in 0..3)
    # W: 16x49 (rows: k*4+i for k,i in 0..3)
    
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    
    term_idx = 0
    for t in range(7):  # top-level Strassen term
        for s in range(7):  # bottom-level Strassen term
            # Top-level factors
            for I in range(2):  # block row of A
                for J in range(2):  # block col of A
                    u_top = top_U[I*2+J][t]
                    if u_top == 0:
                        continue
                    for i_local in range(2):
                        for j_local in range(2):
                            u_bot = U_strassen[i_local*2+j_local][s]
                            if u_bot == 0:
                                continue
                            i = 2*I + i_local
                            j = 2*J + j_local
                            U_444[i*4+j][term_idx] += u_top * u_bot
            
            for J in range(2):  # block col of A / block row of B
                for K in range(2):  # block col of B
                    v_top = top_V[J*2+K][t]
                    if v_top == 0:
                        continue
                    for j_local in range(2):
                        for k_local in range(2):
                            v_bot = V_strassen[j_local*2+k_local][s]
                            if v_bot == 0:
                                continue
                            j = 2*J + j_local
                            k = 2*K + k_local
                            V_444[j*4+k][term_idx] += v_top * v_bot
            
            for K in range(2):  # block col of B
                for I in range(2):  # block row of C
                    w_top = top_W[K*2+I][t]
                    if w_top == 0:
                        continue
                    for k_local in range(2):
                        for i_local in range(2):
                            w_bot = W_strassen[k_local*2+i_local][s]
                            if w_bot == 0:
                                continue
                            k = 2*K + k_local
                            i = 2*I + i_local
                            W_444[k*4+i][term_idx] += w_top * w_bot
            
            term_idx += 1
    
    # Now add the 49 terms for 4x4x4 to the main matrices (columns 0-48)
    for i in range(16):
        for col in range(49):
            U[i][col] = U_444[i][col]
    
    # For V, the 4x4x4 part uses columns 0-3 of B (j*4+k where k in 0..3)
    # V row index for 4x4x4: j*4+k where j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            for col in range(49):
                V[j*5+k][col] = V_444[j*4+k][col]
    
    # For W, the 4x4x4 part uses k in 0..3, i in 0..3
    # W row index: k*4+i where k in 0..3, i in 0..3
    for k in range(4):
        for i in range(4):
            for col in range(49):
                W[k*4+i][col] = W_444[k*4+i][col]
    
    # Now add 16 scalar terms for the 5th column of B (k=4)
    # We need to compute C[:,4] = A @ B[:,4]
    # This is 4x4 scalar multiplications
    # For each (i,j) in 0..3 x 0..3:
    #   U[i*4+j, 49+i*4+j] = 1
    #   V[j*5+4, 49+i*4+j] = 1  (j is col of A, 4 is col of B)
    #   W[4*4+i, 49+i*4+j] = 1  (4 is col of B, i is row of C)
    
    for i in range(4):
        for j in range(4):
            col = 49 + i*4 + j
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
    
    # Verify time
    if time.time() > deadline:
        # Return what we have (should be complete)
        pass
    
    return {"U": U, "V": V, "W": W}
