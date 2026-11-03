"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # We construct the rank 65 decomposition.
    # Part 1: Strassen for 4x4 x 4x4 (rank 49)
    # Part 2: Schoolbook for 4x4 x 4x1 (rank 16)
    
    rank_total = 65
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4 x 4x4 ---
    
    # Strassen 2x2 terms
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, -1, -1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]),
    ]
    
    # Outer terms for 4x4
    # U_blocks, V_blocks, W_blocks
    # Indices: 0,1,2,3 correspond to blocks 11, 12, 21, 22
    outer_terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, -1]]),
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 1], [0, 0]]),
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 1]]),
    ]
    
    term_index = 0
    for t_out in range(7):
        if time.time() > deadline: break
        U_blocks, V_blocks, W_blocks = outer_terms[t_out]
        
        for t_in in range(7):
            if time.time() > deadline: break
            U_in, V_in, W_in = strassen_2x2[t_in]
            
            # Construct U_full, V_full, W_full
            U_full = [0] * 16
            V_full = [0] * 16
            W_full = [0] * 16
            
            # Map inner indices to outer blocks
            # U_in[k] corresponds to X_entry k
            # X_entry k is a 2x2 block of X
            # X is formed from A blocks using U_blocks
            
            for k in range(4):
                coeff_u = U_in[k]
                coeff_v = V_in[k]
                coeff_w = W_in[k]
                
                if coeff_u == 0 and coeff_v == 0 and coeff_w == 0:
                    continue
                    
                # k corresponds to block index in 2x2 super-matrix
                # k=0 -> (0,0), k=1 -> (0,1), k=2 -> (1,0), k=3 -> (1,1)
                bi, bj = divmod(k, 2)
                
                # U_blocks[bi, bj] is the coefficient for A_block[bi, bj]
                # Wait, U_blocks is 2x2.
                # For P1: U_blocks = [[1, 0], [0, 1]].
                # This means X[0,0] (block 0,0) gets 1 * A11.
                # X[0,1] (block 0,1) gets 0.
                # So we multiply coeff_u by U_blocks[bi, bj].
                
                a_coeff = coeff_u * U_blocks[bi][bj]
                b_coeff = coeff_v * V_blocks[bi][bj]
                c_coeff = coeff_w * W_blocks[bi][bj]
                
                if a_coeff != 0:
                    # Add to U_full
                    # A_block[bi, bj] is a 2x2 matrix
                    # Indices in A:
                    # Block (0,0): A[0,0], A[0,1], A[1,0], A[1,1]
                    # Block (0,1): A[0,2], A[0,3], A[1,2], A[1,3]
                    # Block (1,0): A[2,0], A[2,1], A[3,0], A[3,1]
                    # Block (1,1): A[2,2], A[2,3], A[3,2], A[3,3]
                    base_i = 2 * bi
                    base_j = 2 * bj
                    for ii in range(2):
                        for jj in range(2):
                            idx = (base_i + ii) * 4 + (base_j + jj)
                            U_full[idx] += a_coeff
                            
                if b_coeff != 0:
                    # Add to V_full
                    # B_block[bi, bj]
                    base_i = 2 * bi
                    base_j = 2 * bj
                    for ii in range(2):
                        for jj in range(2):
                            idx = (base_i + ii) * 4 + (base_j + jj)
                            V_full[idx] += b_coeff
                            
                if c_coeff != 0:
                    # Add to W_full
                    # C_block[bi, bj]
                    base_i = 2 * bi
                    base_j = 2 * bj
                    for ii in range(2):
                        for jj in range(2):
                            # W index: k*n + i
                            # k is column of C (0..3)
                            # i is row of C (0..3)
                            # C[k, i]
                            # Block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                            # So k (col) = 2*bj + jj
                            # i (row) = 2*bi + ii
                            idx = (2 * bj + jj) * 4 + (2 * bi + ii)
                            W_full[idx] += c_coeff

            # Assign to U, V, W
            for i in range(16):
                if U_full[i] != 0:
                    U[i][term_index] = U_full[i]
                if V_full[i] != 0:
                    V[i][term_index] = V_full[i]
                if W_full[i] != 0:
                    W[i][term_index] = W_full[i]
            
            term_index += 1

    # --- Part 2: Schoolbook for 4x4 x 4x1 ---
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
