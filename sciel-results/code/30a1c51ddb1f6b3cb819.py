"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: 2x2 block structure of 3x3 matrices with Strassen (rank 7)
    # for the block level, and Bürgisser rank-23 for each 3x3 product.
    # Total rank = 7 * 23 = 161
    
    rank = 161
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 161)
    V = [[0] * rank for _ in range(m * p)]  # (36, 161)
    W = [[0] * rank for _ in range(p * n)]  # (36, 161)
    
    # Bürgisser rank-23 decomposition for 3x3 matrix multiplication
    # A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    # C = A*B
    
    # 23 bilinear forms: each term t has:
    #   u_terms: list of (a_index, coeff) for linear combo of A entries
    #   v_terms: list of (b_index, coeff) for linear combo of B entries
    #   w_terms: list of (c_index, coeff) for contributions to C entries
    # C indices: c0=C[0,0], c1=C[0,1], c2=C[0,2], c3=C[1,0], c4=C[1,1], c5=C[1,2], c6=C[2,0], c7=C[2,1], c8=C[2,2]
    
    # Using a verified rank-23 decomposition (Bürgisser 2013, simplified presentation)
    # This uses the "23-term" algorithm with specific coefficients.
    
    burgisser_terms = [
        # Term 0: (a0+a8)(b0+b8)
        ([(0,1),(8,1)], [(0,1),(8,1)], [(0,1),(8,1)]),
        # Term 1: (a1+a8)(b1+b8)  
        ([(1,1),(8,1)], [(1,1),(8,1)], [(1,1),(8,1)]),
        # Term 2: (a2+a8)(b2+b8)
        ([(2,1),(8,1)], [(2,1),(8,1)], [(2,1),(8,1)]),
        # Term 3: (a3+a8)(b3+b8)
        ([(3,1),(8,1)], [(3,1),(8,1)], [(3,1),(8,1)]),
        # Term 4: (a4+a8)(b4+b8)
        ([(4,1),(8,1)], [(4,1),(8,1)], [(4,1),(8,1)]),
        # Term 5: (a5+a8)(b5+b8)
        ([(5,1),(8,1)], [(5,1),(8,1)], [(5,1),(8,1)]),
        # Term 6: (a6+a8)(b6+b8)
        ([(6,1),(8,1)], [(6,1),(8,1)], [(6,1),(8,1)]),
        # Term 7: (a7+a8)(b7+b8)
        ([(7,1),(8,1)], [(7,1),(8,1)], [(7,1),(8,1)]),
        # Term 8: a0*b0
        ([(0,1)], [(0,1)], [(0,1)]),
        # Term 9: a1*b1
        ([(1,1)], [(1,1)], [(1,1)]),
        # Term 10: a2*b2
        ([(2,1)], [(2,1)], [(2,1)]),
        # Term 11: a3*b3
        ([(3,1)], [(3,1)], [(3,1)]),
        # Term 12: a4*b4
        ([(4,1)], [(4,1)], [(4,1)]),
        # Term 13: a5*b5
        ([(5,1)], [(5,1)], [(5,1)]),
        # Term 14: a6*b6
        ([(6,1)], [(6,1)], [(6,1)]),
        # Term 15: a7*b7
        ([(7,1)], [(7,1)], [(7,1)]),
        # Term 16: (a0-a8)(b8)
        ([(0,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 17: (a1-a8)(b8)
        ([(1,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 18: (a2-a8)(b8)
        ([(2,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 19: (a3-a8)(b8)
        ([(3,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 20: (a4-a8)(b8)
        ([(4,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 21: (a5-a8)(b8)
        ([(5,1),(8,-1)], [(8,1)], [(8,-1)]),
        # Term 22: (a6-a8)(b8)
        ([(6,1),(8,-1)], [(8,1)], [(8,-1)]),
    ]
    
    # Wait - the above is NOT a valid rank-23 decomposition. That's just schoolbook-like.
    # I need the actual Bürgisser rank-23 coefficients.
    
    # Let me use a different, verified approach: 
    # Keep the current 189-rank approach (Strassen 2x2 blocks of 2x2, schoolbook 3x3)
    # but verify it's correct and well-structured.
    
    # Actually, let me use a cleaner formulation of the 189-rank approach.
    # Partition 6x6 into 2x2 blocks of 3x3 matrices.
    # Use Strassen (rank 7) for the 2x2 block structure.
    # Use schoolbook (27) for each 3x3 product.
    # Total = 7 * 27 = 161? No, 7 * 27 = 189.
    
    # Strassen for 2x2 block multiplication:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product, done with schoolbook (27 scalar mults)
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 block structure
    # Each term: (A_comb, B_comb, C_contrib)
    # A_comb: list of (block_id, coeff) where block_id 0=A00,1=A01,2=A10,3=A11
    # B_comb: list of (block_id, coeff) where block_id 0=B00,1=B01,2=B10,3=B11
    # C_contrib: list of (block_id, coeff) where block_id 0=C00,1=C01,2=C10,3=C11
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        (([0,3],[1,1]), ([0,3],[1,1]), [(0,1),(3,1)]),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
        (([2,3],[1,1]), ([0],[1]), [(2,1),(3,-1)]),
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        (([0],[1]), ([1,3],[1,-1]), [(1,1),(3,1)]),
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        (([3],[1]), ([2,0],[1,-1]), [(0,1),(2,1)]),
        # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
        (([0,1],[1,1]), ([3],[1]), [(0,-1),(1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        (([2,0],[1,-1]), ([0,1],[1,1]), [(3,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        (([1,3],[1,-1]), ([2,3],[1,1]), [(0,1)]),
    ]
    
    # Block offsets: block_id -> (row_offset, col_offset) for 3x3 sub-blocks
    # Block 0: rows 0-2, cols 0-2
    # Block 1: rows 0-2, cols 3-5
    # Block 2: rows 3-5, cols 0-2
    # Block 3: rows 3-5, cols 3-5
    block_offsets = {0: (0,0), 1: (0,3), 2: (3,0), 3: (3,3)}
    
    col = 0
    for a_blocks, a_coeffs, b_blocks, b_coeffs, c_contrib in strassen_terms:
        # For each of the 27 scalar multiplications in the 3x3 product
        for i in range(3):  # row in 3x3
            for k in range(3):  # col in A / row in B
                for j in range(3):  # col in B / col in C
                    c_idx = col
                    col += 1
                    
                    # U factors: A elements
                    for bid, coeff in zip(a_blocks, a_coeffs):
                        r_off, c_off = block_offsets[bid]
                        global_row = r_off + i
                        global_col = c_off + k
                        U[global_row * 6 + global_col][c_idx] += coeff
                    
                    # V factors: B elements
                    # V index: row_B * 6 + col_B
                    # B block bid: row offset r_off, col offset c_off
                    # B element at (k, j) within block -> global (r_off+k, c_off+j)
                    for bid, coeff in zip(b_blocks, b_coeffs):
                        r_off, c_off = block_offsets[bid]
                        global_row = r_off + k
                        global_col = c_off + j
                        V[global_row * 6 + global_col][c_idx] += coeff
                    
                    # W factors: C elements
                    # W convention: W[col_C * n + row_C, r] -> W index = col_C * 6 + row_C
                    # C element at (i, j) within block -> global (r_off+i, c_off+j)
                    # W index = (c_off+j) * 6 + (r_off+i)
                    for bid, coeff in c_contrib:
                        r_off, c_off = block_offsets[bid]
                        global_row = r_off + i
                        global_col = c_off + j
                        W[global_col * 6 + global_row][c_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
