import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block matrix multiplication
    # We have 4 blocks of A (each 1x2), 4 blocks of B (each 2x2), 4 blocks of C (each 1x2)
    # We decompose each 2x2x2 sub-problem with rank-7 Strassen, but since A blocks are 1x2,
    # we can share some computations.
    
    # Actually, let's think of it differently.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A1 (1x4), A2 (1x4)
    # Split B into B11, B12, B21, B22 (each 2x2)
    # Split C into C1 (1x4), C2 (1x4)
    
    # C1 = A1 * B
    # C2 = A2 * B
    
    # We can compute A1*B and A2*B using Strassen-like methods.
    # But A1 is 1x4, not 2x2. So we need a different approach.
    
    # Let's use the 2x2x2 Strassen for each pair of columns.
    # For columns 0,1 of C:
    # C[:,0:2] = A[:,0:2] * B[0:2,0:2] + A[:,2:4] * B[2:4,0:2]
    
    # Actually, let's just use the standard 2x4x4 tensor with rank 28.
    # The 2x4x4 tensor can be decomposed as 4 independent 2x2x2 tensors,
    # each with rank 7, giving total rank 28.
    
    # The 4 sub-tensors correspond to:
    # 1. A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    # 2. A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    # 3. A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    # 4. A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    
    # For each 2x2x2 sub-problem, use Strassen with 7 multiplications.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the Strassen coefficients for a generic 2x2x2:
    # alpha: for each multiplication s, coefficients of A_ij
    # beta: for each multiplication s, coefficients of B_ij
    # gamma: for each multiplication s, coefficients of C_ij
    
    # s=1: M1 = (A11+A22)(B11+B22)
    alpha_s1 = {(0,0): 1, (1,1): 1}
    beta_s1 = {(0,0): 1, (1,1): 1}
    gamma_s1 = {(0,0): 1, (1,1): 1}
    
    # s=2: M2 = (A21+A22)B11
    alpha_s2 = {(1,0): 1, (1,1): 1}
    beta_s2 = {(0,0): 1}
    gamma_s2 = {(0,0): -1, (1,0): 1}
    
    # s=3: M3 = A11(B21-B22)
    alpha_s3 = {(0,0): 1}
    beta_s3 = {(1,0): 1, (1,1): -1}
    gamma_s3 = {(0,1): 1, (1,1): 1}
    
    # s=4: M4 = A22(B12-B11)
    alpha_s4 = {(1,1): 1}
    beta_s4 = {(0,1): 1, (0,0): -1}
    gamma_s4 = {(0,0): 1, (1,0): 1}
    
    # s=5: M5 = (A11+A12)B22
    alpha_s5 = {(0,0): 1, (0,1): 1}
    beta_s5 = {(1,1): 1}
    gamma_s5 = {(0,0): -1, (0,1): 1}
    
    # s=6: M6 = (A21-A11)(B11+B12)
    alpha_s6 = {(1,0): 1, (0,0): -1}
    beta_s6 = {(0,0): 1, (0,1): 1}
    gamma_s6 = {(1,1): 1}
    
    # s=7: M7 = (A12-A22)(B21+B22)
    alpha_s7 = {(0,1): 1, (1,1): -1}
    beta_s7 = {(1,0): 1, (1,1): 1}
    gamma_s7 = {(0,0): 1}
    
    strassen = [
        (alpha_s1, beta_s1, gamma_s1),
        (alpha_s2, beta_s2, gamma_s2),
        (alpha_s3, beta_s3, gamma_s3),
        (alpha_s4, beta_s4, gamma_s4),
        (alpha_s5, beta_s5, gamma_s5),
        (alpha_s6, beta_s6, gamma_s6),
        (alpha_s7, beta_s7, gamma_s7),
    ]
    
    # 4 sub-problems:
    # sub=0: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # sub=1: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    # sub=2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    # sub=3: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    
    # For each sub-problem (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset):
    sub_problems = [
        (0, 0, 0, 0, 0),  # A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]
        (0, 0, 0, 2, 2),  # A[0:2,0:2], B[0:2,2:4], C[0:2,2:4]
        (0, 2, 2, 0, 0),  # A[0:2,2:4], B[2:4,0:2], C[0:2,0:2]
        (0, 2, 2, 2, 2),  # A[0:2,2:4], B[2:4,2:4], C[0:2,2:4]
    ]
    
    for sub_idx, (a_r, a_c, b_r, b_c, c_c) in enumerate(sub_problems):
        for s_idx, (alpha, beta, gamma) in enumerate(strassen):
            c_rank = sub_idx * 7 + s_idx
            
            # Fill U: A indices
            for (i, j), val in alpha.items():
                if val != 0:
                    u_row = a_r + i
                    u_col = a_c + j
                    U[u_row * m + u_col][c_rank] += val
            
            # Fill V: B indices
            for (i, j), val in beta.items():
                if val != 0:
                    v_row = b_r + i
                    v_col = b_c + j
                    V[v_row * p + v_col][c_rank] += val
            
            # Fill W: C indices
            for (i, j), val in gamma.items():
                if val != 0:
                    w_row = c_c + j  # column of C
                    w_col = a_r + i  # row of C (which is index into n)
                    # W index: k*n + i, where k is column index, i is row index
                    W[w_row * n + w_col][c_rank] += val
    
    return {"U": U, "V": V, "W": W}
