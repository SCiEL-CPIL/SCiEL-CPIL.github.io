import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    
    The decomposition uses block partitioning:
    - Two M_2,4,4 blocks (each of rank 28) = 56
    - Two M_1,4,4 blocks (each of rank 16) = 32
    Total rank: 88
    """
    n, m, p = DIMENSIONS
    
    # Rank 88 decomposition
    rank = 88
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (n*m, rank) = (12, 88)
    V = [[0] * rank for _ in range(m * p)]  # shape (m*p, rank) = (32, 88)
    W = [[0] * rank for _ in range(p * n)]  # shape (p*n, rank) = (24, 88)
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    strassen_factors = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A21+A22)B11 -> C21 += 1, C22 -= 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: A11(B12-B22) -> C12 += 1, C22 += 1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: A22(B21-B11) -> C11 += 1, C21 += 1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        # For each c_block, we have:
        # A is 2x4, B is 4x4, C is 2x4
        # The matrix product is C = A @ B
        
        # We decompose this using 4 sub-products:
        # 1. A[:, 0:2] @ B[0:2, :] -> contributes to C
        # 2. A[:, 2:4] @ B[2:4, :] -> contributes to C
        
        # Actually, let's use the standard block decomposition:
        # Split A into A_L (2x2) and A_R (2x2)
        # Split B into B_LL (2x2), B_LR (2x2), B_RL (2x2), B_RR (2x2)
        # Then C = A_L @ [B_LL | B_LR] + A_R @ [B_RL | B_RR]
        
        # We'll apply Strassen to each of the four 2x2x2 sub-products:
        # Sub-product 1: A_L (A[0:2, 0:2]) @ B_LL (B[0:2, 0:2]) -> C_L (C[:, 0:2])
        # Sub-product 2: A_L (A[0:2, 0:2]) @ B_LR (B[0:2, 2:4]) -> C_R (C[:, 2:4])
        # Sub-product 3: A_R (A[0:2, 2:4]) @ B_RL (B[2:4, 0:2]) -> C_L (C[:, 0:2])
        # Sub-product 4: A_R (A[0:2, 2:4]) @ B_RR (B[2:4, 2:4]) -> C_R (C[:, 2:4])
        
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            for sub_prod_idx in range(4):
                # Determine offsets for A, B, C
                if sub_prod_idx == 0:
                    # A_L @ B_LL -> C_L
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    # A_L @ B_LR -> C_R
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    # A_R @ B_RL -> C_L
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:  # sub_prod_idx == 3
                    # A_R @ B_RR -> C_R
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                # Apply the Strassen factor to this sub-product
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        # A is 1x4, B is 4x4, C is 1x4
        # C_j = sum_k A_k B_kj
        # 16 rank-1 terms: A_k * B_kj * C_j
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k  # Row index 2, column k
                v_idx = k * 8 + (j + 4 * c_block)  # Row k, column j + 4*c_block
                w_idx = (j + 4 * c_block) * 3 + 2  # Row j+4*c_block, column 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
