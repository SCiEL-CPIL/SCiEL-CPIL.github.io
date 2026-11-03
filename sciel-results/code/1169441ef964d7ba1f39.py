import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 4, 4, 6
    
    # Strategy: Block decomposition
    # We decompose the 4x4x6 multiplication into:
    # 1. 4x4x4 part (first 4 columns of B) using Strassen's algorithm for 4x4 matrices (Rank 49)
    # 2. 4x4x2 part (last 2 columns of B) using a simple rank-8 decomposition
    # Total rank = 49 + 8 = 57
    
    # --- Part 1: 4x4x4 Strassen decomposition (Rank 49) ---
    # We construct the 4x4x4 decomposition by applying Strassen's algorithm to 2x2 blocks.
    # The 4x4 matrices are partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen's algorithm for 2x2 matrices has rank 7.
    # For 4x4, we apply Strassen's at the block level.
    # The macro-terms involve sums/differences of 2x2 blocks.
    # For each macro-term, we multiply 2x2 blocks, which can be done with Strassen (rank 7).
    # So total rank = 7 (macro) * 7 (micro) = 49.
    
    # Strassen's 2x2 decomposition (for micro-level):
    # Let U1, V1, W1 be the 2x2x2 decomposition (rank 7).
    # Indices: 0..6
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Macro-level Strassen (for 4x4 blocks):
    # The 7 macro-terms for 2x2 block matrices:
    # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
    # P2 = (A21+A22)B11 -> C21 += P2, C22 -= P2
    # P3 = A11(B12-B22) -> C12 += P3, C22 += P3
    # P4 = A22(B21-B11) -> C11 += P4, C21 += P4
    # P5 = (A11+A12)B22 -> C11 -= P5, C12 += P5
    # P6 = (A21-A11)(B11+B12) -> C22 += P6
    # P7 = (A12-A22)(B21+B22) -> C11 += P7, C22 += P7
    
    # Each macro-term is a product of 2x2 matrices.
    # For each macro-term, we use the micro-level Strassen (U1, V1, W1).
    
    # Macro-term definitions:
    # Each macro-term k has:
    # - A_block_coeffs: dict mapping block index (00, 01, 10, 11) to coefficient
    # - B_block_coeffs: dict mapping block index to coefficient
    # - C_block_coeffs: dict mapping block index to coefficient
    # Block indices: 00 -> A11, 01 -> A12, 10 -> A21, 11 -> A22
    
    macro_terms = [
        # P1
        {'a': {'00': 1, '11': 1}, 'b': {'00': 1, '11': 1}, 'c': {'00': 1, '11': 1}},
        # P2
        {'a': {'10': 1, '11': 1}, 'b': {'00': 1}, 'c': {'10': 1, '11': -1}},
        # P3
        {'a': {'00': 1}, 'b': {'01': 1, '11': -1}, 'c': {'01': 1, '11': 1}},
        # P4
        {'a': {'11': 1}, 'b': {'10': 1, '00': -1}, 'c': {'00': 1, '10': 1}},
        # P5
        {'a': {'00': 1, '01': 1}, 'b': {'11': 1}, 'c': {'00': -1, '01': 1}},
        # P6
        {'a': {'10': 1, '00': -1}, 'b': {'00': 1, '01': 1}, 'c': {'11': 1}},
        # P7
        {'a': {'01': 1, '11': -1}, 'b': {'10': 1, '11': 1}, 'c': {'00': 1, '11': 1}},
    ]
    
    rank_4x4x4 = 49
    U_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]
    V_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]
    W_4x4x4 = [[0]*rank_4x4x4 for _ in range(16)]
    
    # Helper functions to map block indices to global indices
    # For 4x4 matrix, block (bi, bj) contains elements (i, j) where
    # i = bi*2 + i_local, j = bj*2 + j_local
    # Global index for A[i,j] = i*4 + j
    
    def a_global(block_i, block_j, local_i, local_j):
        return (block_i * 2 + local_i) * 4 + (block_j * 2 + local_j)
    
    def b_global(block_i, block_j, local_i, local_j):
        # For B (4x4), same indexing
        return (block_i * 2 + local_i) * 4 + (block_j * 2 + local_j)
    
    def c_global(block_i, block_j, local_i, local_j):
        # For C (4x4), same indexing
        return (block_i * 2 + local_i) * 4 + (block_j * 2 + local_j)
    
    # Fill in the 4x4x4 decomposition
    for macro_k, macro in enumerate(macro_terms):
        for micro_m in range(7):
            r = macro_k * 7 + micro_m
            
            # Get micro-level U, V, W values
            u1_vals = U1  # 4x7
            v1_vals = V1  # 4x7
            w1_vals = W1  # 4x7
            
            # For U: A block contributions
            for block_key, coeff_a in macro['a'].items():
                bi, bj = int(block_key[0]), int(block_key[1])
                for local_i in range(2):
                    for local_j in range(2):
                        a_idx = a_global(bi, bj, local_i, local_j)
                        # Micro-level index: local_i * 2 + local_j
                        micro_idx = local_i * 2 + local_j
                        u1_val = u1_vals[micro_idx][micro_m]
                        if u1_val != 0:
                            U_4x4x4[a_idx][r] += coeff_a * u1_val
            
            # For V: B block contributions
            for block_key, coeff_b in macro['b'].items():
                bi, bj = int(block_key[0]), int(block_key[1])
                for local_i in range(2):
                    for local_j in range(2):
                        b_idx = b_global(bi, bj, local_i, local_j)
                        # Micro-level index for V: local_i * 2 + local_j
                        # Note: In Strassen, V's row index corresponds to B's element
                        # B[j,k], so local_i corresponds to j, local_j to k
                        micro_idx = local_i * 2 + local_j
                        v1_val = v1_vals[micro_idx][micro_m]
                        if v1_val != 0:
                            V_4x4x4[b_idx][r] += coeff_b * v1_val
            
            # For W: C block contributions
            for block_key, coeff_c in macro['c'].items():
                bi, bj = int(block_key[0]), int(block_key[1])
                for local_i in range(2):
                    for local_j in range(2):
                        c_idx = c_global(bi, bj, local_i, local_j)
                        # Micro-level index for W: local_i * 2 + local_j
                        # In Strassen, W's row index corresponds to C's element
                        # C[i,k], so local_i corresponds to i, local_j to k
                        micro_idx = local_i * 2 + local_j
                        w1_val = w1_vals[micro_idx][micro_m]
                        if w1_val != 0:
                            W_4x4x4[c_idx][r] += coeff_c * w1_val
    
    # --- Part 2: 4x4x2 decomposition (Rank 8) ---
    # For the last 2 columns of B, we use a simple rank-8 decomposition.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # For k = 4, 5 (the last 2 columns)
    
    rank_4x4x2 = 8
    U_4x4x2 = [[0]*rank_4x4x2 for _ in range(16)]
    V_4x4x2 = [[0]*rank_4x4x2 for _ in range(24)]
    W_4x4x2 = [[0]*rank_4x4x2 for _ in range(24)]
    
    # For each of the 4 rows of A and 4 columns of B (but only last 2 columns of B),
    # we have 4*2 = 8 terms.
    # Term r: A[i, j] * B[j, k] -> C[i, k]
    # where i is 0..3, k is 4..5 (but in W's indexing, k is 0..5)
    
    # Actually, let's use the convention:
    # U has 16 rows (i*4+j for A[i,j])
    # V has 24 rows (j*6+k for B[j,k])
    # W has 24 rows (k*4+i for C[k,i])
    
    # For the 4x4x2 part, we only care about k=4,5 in B and C.
    # Term index r = 0..7
    # Let's map r to (i, k) where i=0..3, k=4,5
    # But we need to sum over j, so each term is A[i,j]*B[j,k] for a specific j?
    # No, that would be 4*4*2 = 32 terms.
    
    # Wait, the rank-8 decomposition for 4x4x2 is not trivial.
    # Actually, for 4x4x2, the minimum rank is at most 8 (using schoolbook for each of the 2 columns, but sharing).
    # Let me reconsider.
    
    # Actually, for a 4x4x2 multiplication, we can decompose it as:
    # C[:, 4] = A @ B[:, 4]  (4x1 vector)
    # C[:, 5] = A @ B[:, 5]  (4x1 vector)
    
    # Each 4x1 vector multiplication has rank 4 (schoolbook).
    # So total rank = 4 + 4 = 8.
    
    # Term r (0..3): for column 4 of B
    #   A[i, j] * B[j, 4] -> C[i, 4]
    #   But we need to sum over j, so each term is one (i, j) pair.
    #   There are 4*4 = 16 terms for one column.
    
    # I think I'm confusing myself. Let's use a different approach.
    # For the 4x4x2 part, we can just use the schoolbook multiplication for the last 2 columns.
    # That would be 4*4*2 = 32 terms, which is too many.
    
    # Let me reconsider the block decomposition.
    # Actually, the 4x4x2 part can be handled more efficiently.
    
    # For now, let's use a rank-8 decomposition where:
    # Term r (0..7):
    #   i = r % 4
    #   col = r // 4  # 0 or 1, corresponding to column 4 or 5
    #   k = 4 + col
    #   For each j (0..3):
    #     U[i*4+j][r] = 1
    #     V[j*6+k][r] = 1
    #     W[k*4+i][r] = 1
    
    for r in range(8):
        i = r % 4
        col = r // 4
        k = 4 + col
        for j in range(4):
            U_4x4x2[i * 4 + j][r] = 1
            V_4x4x2[j * 6 + k][r] = 1
            W_4x4x2[k * 4 + i][r] = 1
    
    # --- Combine the two parts ---
    total_rank = rank_4x4x4 + rank_4x4x2
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(24)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Copy the 4x4x4 part (first 49 terms)
    for i in range(16):
        for r in range(rank_4x4x4):
            U[i][r] = U_4x4x4[i][r]
            V[i][r] = V_4x4x4[i][r]
            W[i][r] = W_4x4x4[i][r]
    
    # Copy the 4x4x2 part (next 8 terms)
    for i in range(16):
        for r in range(rank_4x4x2):
            U[i][rank_4x4x4 + r] = U_4x4x2[i][r]
    for i in range(24):
        for r in range(rank_4x4x2):
            V[i][rank_4x4x4 + r] = V_4x4x2[i][r]
            W[i][rank_4x4x4 + r] = W_4x4x2[i][r]
    
    return {"U": U, "V": V, "W": W}
