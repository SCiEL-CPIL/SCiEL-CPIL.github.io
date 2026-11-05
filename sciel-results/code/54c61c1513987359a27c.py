"""Matrix multiplication 6x6x6 tensor decomposition using Strassen 2x2 blocks with 3x3 schoolbook."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    
    Strategy: 2x2 block decomposition with 3x3 blocks, using Strassen's 7-product algorithm
    at the outer block level and schoolbook for each 3x3 block multiplication.
    Total rank = 7 * 27 = 189 (better than schoolbook 216).
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6
    
    # Block structure: 6x6 = 2x2 blocks of 3x3
    block_size = 3
    num_blocks = 2
    
    # Strassen formulas for 2x2 block multiplication:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
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
    
    strassen_blocks = [
        # (A block1, A block2, B block1, B block2, sign_A1, sign_A2, sign_B1, sign_B2, 
        #  C block1, C block2, C block3, C block4, sign_C1, sign_C2, sign_C3, sign_C4)
        # M1 = (A11+A22)*(B11+B22)
        ((0,0), (1,1), (0,0), (1,1), 1, 1, 1, 1,
         (0,0), (1,1), None, None, 1, 1, 0, 0),  # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)*B11
        ((1,0), (1,1), (0,0), None, 1, 1, 1, 0,
         None, (1,0), None, (1,1), 0, 1, 0, -1),  # C21 += M2, C22 -= M2
        
        # M3 = A11*(B12-B22)
        ((0,0), None, (0,1), (1,1), 1, 0, 1, -1,
         None, (0,1), None, (1,1), 0, 1, 0, 1),  # C12 += M3, C22 += M3
        
        # M4 = A22*(B21-B11)
        ((1,1), None, (1,0), (0,0), 1, 0, 1, -1,
         (0,0), None, (1,0), None, 1, 0, 1, 0),  # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)*B22
        ((0,0), (0,1), (1,1), None, 1, 1, 1, 0,
         (0,0), None, (0,1), None, -1, 0, 1, 0),  # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)*(B11+B12)
        ((1,0), (0,0), (0,0), (0,1), 1, -1, 1, 1,
         None, None, None, (1,1), 0, 0, 0, 1),  # C22 += M6
        
        # M7 = (A12-A22)*(B21+B22)
        ((0,1), (1,1), (1,0), (1,1), 1, -1, 1, 1,
         (0,0), None, None, None, 1, 0, 0, 0),  # C11 += M7
    ]
    
    # Total rank: 7 Strassen steps * 27 schoolbook terms per 3x3 block
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    col = 0
    
    for strassen_entry in strassen_blocks:
        a_block1, a_block2, b_block1, b_block2 = strassen_entry[0], strassen_entry[1], strassen_entry[2], strassen_entry[3]
        sign_a1, sign_a2, sign_b1, sign_b2 = strassen_entry[4], strassen_entry[5], strassen_entry[6], strassen_entry[7]
        c_blocks = strassen_entry[8:12]
        c_signs = strassen_entry[12:16]
        
        # For each of the 27 schoolbook terms in the 3x3 block multiplication
        for i_local in range(block_size):  # row of X
            for j_local in range(block_size):  # col of X / row of Y
                for k_local in range(block_size):  # col of Y
                    # X[i_local, j_local] * Y[j_local, k_local] -> Z[i_local, k_local]
                    
                    # Update U (for A matrix)
                    # X[i,j] = sum of A_block entries
                    if a_block1 is not None:
                        u_idx = (a_block1[0] * block_size + i_local) * m + (a_block1[1] * block_size + j_local)
                        U[u_idx][col] += sign_a1
                    
                    if a_block2 is not None:
                        u_idx = (a_block2[0] * block_size + i_local) * m + (a_block2[1] * block_size + j_local)
                        U[u_idx][col] += sign_a2
                    
                    # Update V (for B matrix)
                    # Y[j,k] = sum of B_block entries
                    if b_block1 is not None:
                        v_idx = (b_block1[0] * block_size + j_local) * p + (b_block1[1] * block_size + k_local)
                        V[v_idx][col] += sign_b1
                    
                    if b_block2 is not None:
                        v_idx = (b_block2[0] * block_size + j_local) * p + (b_block2[1] * block_size + k_local)
                        V[v_idx][col] += sign_b2
                    
                    # Update W (for C matrix)
                    # Z[i,k] contributes to C blocks
                    for ci in range(4):
                        if c_blocks[ci] is not None:
                            # W[k*n+i, r] convention: C[i,k] is at W[k*n+i]
                            w_idx = (c_blocks[ci][1] * block_size + k_local) * n + (c_blocks[ci][0] * block_size + i_local)
                            W[w_idx][col] += c_signs[ci]
                    
                    col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook 6x6x6 (rank 216)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
