import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen's algorithm
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Define the 8 instances
    # Each instance is a tuple of:
    # (a_offsets, b_offsets, c_offsets)
    # a_offsets: list of 4 (i,j) pairs for the 2x2 submatrix of A
    # b_offsets: list of 4 (j,k) pairs for the 2x2 submatrix of B
    # c_offsets: list of 4 (i,k) pairs for the 2x2 submatrix of C
    
    # A is 2x4. Indices (i,j) with i in 0,1 and j in 0..3.
    # B is 4x8. Indices (j,k) with j in 0..3 and k in 0..7.
    # C is 2x8. Indices (i,k) with i in 0,1 and k in 0..7.
    
    # Instance 0: A1 (rows 0-1, cols 0-1) * B_L_top_left (rows 0-1, cols 0-1) -> C_L_top_left (rows 0-1, cols 0-1)
    # A1: (0,0), (0,1), (1,0), (1,1)
    # B_L_top_left: (0,0), (0,1), (1,0), (1,1)
    # C_L_top_left: (0,0), (0,1), (1,0), (1,1)
    inst_0 = (
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,0), (0,1), (1,0), (1,1)]
    )
    
    # Instance 1: A1 * B_L_top_right (rows 0-1, cols 2-3) -> C_L_top_right (rows 0-1, cols 2-3)
    inst_1 = (
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,2), (0,3), (1,2), (1,3)],
        [(0,2), (0,3), (1,2), (1,3)]
    )
    
    # Instance 2: A2 (rows 0-1, cols 2-3) * B_L_bot_left (rows 2-3, cols 0-1) -> C_L_bot_left (rows 0-1, cols 0-1)
    # Note: A2 is cols 2-3 of A. B_L_bot is rows 2-3 of B.
    # Result adds to C rows 0-1, cols 0-1.
    inst_2 = (
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,0), (2,1), (3,0), (3,1)],
        [(0,0), (0,1), (1,0), (1,1)]
    )
    
    # Instance 3: A2 * B_L_bot_right (rows 2-3, cols 2-3) -> C_L_bot_right (rows 0-1, cols 2-3)
    inst_3 = (
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,2), (2,3), (3,2), (3,3)],
        [(0,2), (0,3), (1,2), (1,3)]
    )
    
    # Instance 4: A1 * B_R_top_left (rows 0-1, cols 4-5) -> C_R_top_left (rows 0-1, cols 4-5)
    inst_4 = (
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,4), (0,5), (1,4), (1,5)],
        [(0,4), (0,5), (1,4), (1,5)]
    )
    
    # Instance 5: A1 * B_R_top_right (rows 0-1, cols 6-7) -> C_R_top_right (rows 0-1, cols 6-7)
    inst_5 = (
        [(0,0), (0,1), (1,0), (1,1)],
        [(0,6), (0,7), (1,6), (1,7)],
        [(0,6), (0,7), (1,6), (1,7)]
    )
    
    # Instance 6: A2 * B_R_bot_left (rows 2-3, cols 4-5) -> C_R_bot_left (rows 0-1, cols 4-5)
    inst_6 = (
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,4), (2,5), (3,4), (3,5)],
        [(0,4), (0,5), (1,4), (1,5)]
    )
    
    # Instance 7: A2 * B_R_bot_right (rows 2-3, cols 6-7) -> C_R_bot_right (rows 0-1, cols 6-7)
    inst_7 = (
        [(0,2), (0,3), (1,2), (1,3)],
        [(2,6), (2,7), (3,6), (3,7)],
        [(0,6), (0,7), (1,6), (1,7)]
    )
    
    instances = [inst_0, inst_1, inst_2, inst_3, inst_4, inst_5, inst_6, inst_7]
    
    # Strassen steps definition
    # Each step is a tuple of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local index (0..3) to coefficient
    # v_coeffs: dict mapping local index (0..3) to coefficient
    # w_coeffs: dict mapping local index (0..3) to coefficient
    
    # Local indices:
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    steps = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Note: W coefficients for M1..M7 need to be mapped to c00, c01, c10, c11
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the W coefficients for each step.
    # Step 0 (M1): c00 += 1, c11 += 1. Local indices: c00=0, c11=3.
    # So w_coeffs for Step 0 should be {0: 1, 3: 1}.
    
    # Step 1 (M2): c10 += 1. Local index c10=2.
    # So w_coeffs for Step 1 should be {2: 1}.
    
    # Step 2 (M3): c01 += 1, c11 += 1. Local indices c01=1, c11=3.
    # So w_coeffs for Step 2 should be {1: 1, 3: 1}.
    
    # Step 3 (M4): c00 += 1, c10 += 1. Local indices c00=0, c10=2.
    # So w_coeffs for Step 3 should be {0: 1, 2: 1}.
    
    # Step 4 (M5): c00 -= 1, c01 += 1. Local indices c00=0, c01=1.
    # So w_coeffs for Step 4 should be {0: -1, 1: 1}.
    
    # Step 5 (M6): c11 += 1. Local index c11=3.
    # So w_coeffs for Step 5 should be {3: 1}.
    
    # Step 6 (M7): c00 += 1. Local index c00=0.
    # So w_coeffs for Step 6 should be {0: 1}.
    
    # My previous steps list had some errors in W coefficients.
    # Let's correct them.
    
    corrected_steps = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Wait, M3: c01 = M3 + M5. c11 = M1 - M2 + M3 + M6.
    # So M3 contributes to c01 (1) and c11 (1).
    # Local indices: c01=1, c11=3.
    # So W for M3 is {1: 1, 3: 1}. Correct.
    
    # M4: c00 = M1 + M4 - M5 + M7. c10 = M2 + M4.
    # So M4 contributes to c00 (1) and c10 (1).
    # Local indices: c00=0, c10=2.
    # So W for M4 is {0: 1, 2: 1}. Correct.
    
    # M5: c00 = ... - M5 ... c01 = M3 + M5.
    # So M5 contributes to c00 (-1) and c01 (1).
    # Local indices: c00=0, c01=1.
    # So W for M5 is {0: -1, 1: 1}. Correct.
    
    # M6: c11 = ... + M6.
    # So M6 contributes to c11 (1).
    # Local indices: c11=3.
    # So W for M6 is {3: 1}. Correct.
    
    # M7: c00 = ... + M7.
    # So M7 contributes to c00 (1).
    # Local indices: c00=0.
    # So W for M7 is {0: 1}. Correct.
    
    # M1: c00 = M1 ... c11 = M1 ...
    # So M1 contributes to c00 (1) and c11 (1).
    # Local indices: c00=0, c11=3.
    # So W for M1 is {0: 1, 3: 1}. Correct.
    
    # M2: c10 = M2 ...
    # So M2 contributes to c10 (1).
    # Local indices: c10=2.
    # So W for M2 is {2: 1}. Correct.
    
    # Now populate U, V, W
    
    col_idx = 0
    for inst in instances:
        a_offsets, b_offsets, c_offsets = inst
        for u_coeffs, v_coeffs, w_coeffs in corrected_steps:
            # U
            for local_idx, coeff in u_coeffs.items():
                i, j = a_offsets[local_idx]
                U[i * m + j][col_idx] += coeff
            
            # V
            for local_idx, coeff in v_coeffs.items():
                j, k = b_offsets[local_idx]
                V[j * p + k][col_idx] += coeff
            
            # W
            for local_idx, coeff in w_coeffs.items():
                i, k = c_offsets[local_idx]
                # W index is k * n + i
                W[k * n + i][col_idx] += coeff
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
