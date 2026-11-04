"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # We will use a rank 49 algorithm for the 4x4x4 part (first 4 columns of B)
    # and a rank 48 schoolbook algorithm for the 4x4x3 part (last 3 columns of B)
    # Total rank = 49 + 48 = 97
    
    rank_444 = 49
    rank_443 = 4 * 4 * 3  # 48
    total_rank = rank_444 + rank_443
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Multiplication (Rank 49) ---
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2 matrices)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Known rank-7 decomposition for 2x2x2
    # u_r, v_r, w_r
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Block Strassen coefficients for 4x4x4
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks in each M_k
    # Order of A blocks: 00, 01, 10, 11
    block_u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    block_v_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    block_w_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col_idx = 0
    for k in range(7):
        # For each block multiplication M_k
        # M_k = (sum_a block_u[k][a] A_a) * (sum_b block_v[k][b] B_b)
        # Expand: sum_{a,b} block_u[k][a] block_v[k][b] (A_a B_b)
        # Each A_a B_b is a 2x2x2 multiplication, decomposed into 7 rank-1 terms
        for a in range(4):
            if block_u_coeffs[k][a] == 0: continue
            for b in range(4):
                if block_v_coeffs[k][b] == 0: continue
                coeff_av = block_u_coeffs[k][a] * block_v_coeffs[k][b]
                
                # A_a is a 2x2 block of A. Index in A:
                # a=0 -> A00 (rows 0-1, cols 0-1)
                # a=1 -> A01 (rows 0-1, cols 2-3)
                # a=2 -> A10 (rows 2-3, cols 0-1)
                # a=3 -> A11 (rows 2-3, cols 2-3)
                row_offset_a = (a // 2) * 2
                col_offset_a = (a % 2) * 2
                
                # B_b is a 2x2 block of B. Index in B:
                # b=0 -> B00 (rows 0-1, cols 0-1)
                # b=1 -> B01 (rows 0-1, cols 2-3)
                # b=2 -> B10 (rows 2-3, cols 0-1)
                # b=3 -> B11 (rows 2-3, cols 2-3)
                row_offset_b = (b // 2) * 2
                col_offset_b = (b % 2) * 2
                
                # The result C block is C_{a_row, b_col}
                # C row offset: (a // 2) * 2
                # C col offset: (b % 2) * 2
                # Note: The product A_a B_b contributes to C_{(a//2), (b%2)}?
                # Let's check indices.
                # A is 4x4. A_a is 2x2.
                # A_a B_b is 2x2.
                # This 2x2 result is placed in C at row block (a//2) and col block (b%2).
                # Wait, B_b col index is b%2. So C col block is b%2.
                # C row block is a//2.
                
                c_row_offset = (a // 2) * 2
                c_col_offset = (b % 2) * 2
                
                # Now decompose the 2x2x2 multiplication A_a B_b = C_sub
                # Using the 7 Strassen vectors s_u, s_v, s_w
                for r in range(7):
                    # u_r, v_r, w_r are for 2x2 matrices
                    # Scale by coeff_av
                    scale = coeff_av
                    
                    # Fill U
                    for i in range(2):
                        for j in range(2):
                            u_idx = i * 2 + j
                            val = s_u[r][u_idx] * scale
                            if val != 0:
                                U[row_offset_a + i][col_idx] += val * 0 # U is indexed by i*m+j?
                                # U shape (n*m, rank). n=4, m=4.
                                # U index: i * m + j.
                                # Here i is row of A (0-3), j is col of A (0-3).
                                # row_offset_a + i is the global row.
                                # col_offset_a + j is the global col.
                                u_flat_idx = (row_offset_a + i) * m + (col_offset_a + j)
                                U[u_flat_idx][col_idx] += val
                                
                    # Fill V
                    for i in range(2):
                        for j in range(2):
                            v_idx = i * 2 + j
                            val = s_v[r][v_idx] * scale
                            if val != 0:
                                # V index: j * p + k.
                                # Here j is row of B (0-3), k is col of B (0-3).
                                # row_offset_b + i is global row of B.
                                # col_offset_b + j is global col of B.
                                v_flat_idx = (row_offset_b + i) * p + (col_offset_b + j)
                                V[v_flat_idx][col_idx] += val
                                
                    # Fill W
                    for i in range(2):
                        for j in range(2):
                            w_idx = i * 2 + j
                            val = s_w[r][w_idx] * scale
                            if val != 0:
                                # W index: k * n + i.
                                # Here k is col of C (0-3), i is row of C (0-3).
                                # c_row_offset + i is global row of C.
                                # c_col_offset + j is global col of C.
                                w_flat_idx = (c_col_offset + j) * n + (c_row_offset + i)
                                W[w_flat_idx][col_idx] += val
                            
                    col_idx += 1
                    
        # Check time
        if time.time() > deadline:
            # Fallback to schoolbook if too slow? Unlikely.
            pass

    # --- Part 2: 4x4x3 Multiplication (Rank 48, Schoolbook) ---
    # Columns 4, 5, 6 of B
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
