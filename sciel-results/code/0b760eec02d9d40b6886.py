"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices: A[a, b], B[c, d], C[a, d]
    # a, c in {0, 1}, b, d in {0, 1}
    
    # Coefficients for A part of each of the 7 products
    # P0: A00 + A11
    # P1: A10 + A11
    # P2: A00
    # P3: A11
    # P4: A00 + A01
    # P5: A10 - A00
    # P6: A01 - A11
    A_coeffs = [
        [[1, 0], [0, 1]],  # P0: A00=1, A11=1 (a,b) -> (0,0)=1, (1,1)=1
        [[0, 0], [1, 1]],  # P1: A10=1, A11=1
        [[1, 0], [0, 0]],  # P2: A00=1
        [[0, 0], [0, 1]],  # P3: A11=1
        [[1, 1], [0, 0]],  # P4: A00=1, A01=1
        [[-1, 0], [1, 0]], # P5: A00=-1, A10=1
        [[0, 1], [0, -1]], # P6: A01=1, A11=-1
    ]
    
    # Coefficients for B part of each of the 7 products
    # P0: B00 + B11
    # P1: B00
    # P2: B01 - B11
    # P3: B10 - B00
    # P4: B11
    # P5: B00 + B01
    # P6: B10 + B11
    B_coeffs = [
        [[1, 0], [0, 1]],  # P0: B00=1, B11=1 (c,d) -> (0,0)=1, (1,1)=1
        [[1, 0], [0, 0]],  # P1: B00=1
        [[0, 1], [0, -1]], # P2: B01=1, B11=-1
        [[-1, 0], [1, 0]], # P3: B00=-1, B10=1
        [[0, 0], [0, 1]],  # P4: B11=1
        [[1, 1], [0, 0]],  # P5: B00=1, B01=1
        [[0, 0], [1, 1]],  # P6: B10=1, B11=1
    ]
    
    # Coefficients for C part (output) from each of the 7 products
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_coeffs = [
        [[1, 0], [0, 0]],  # P0: C00=1
        [[0, 0], [1, 0]],  # P1: C10=1
        [[0, 1], [0, 0]],  # P2: C01=1
        [[1, 0], [0, 0]],  # P3: C00=1
        [[-1, 1], [0, 0]], # P4: C00=-1, C01=1
        [[0, 0], [0, 1]],  # P5: C11=1
        [[1, 0], [0, 0]],  # P6: C00=1
    ]
    # Wait, C11 also has P0=1, P1=-1, P2=1, P5=1.
    # Let's re-verify C11 coefficients in C_coeffs list (which is indexed by product P_r)
    # C_coeffs[r][a][d] is coefficient of P_r in C_{a,d}.
    
    # P0 contributes to C00 (1) and C11 (1)
    # P1 contributes to C10 (1) and C11 (-1)
    # P2 contributes to C01 (1) and C11 (1)
    # P3 contributes to C00 (1) and C10 (1)
    # P4 contributes to C00 (-1) and C01 (1)
    # P5 contributes to C11 (1)
    # P6 contributes to C00 (1)
    
    # Corrected C_coeffs:
    C_coeffs = [
        [[1, 0], [1, 0]],  # P0: C00=1, C11=1
        [[0, 0], [-1, 0]], # P1: C10=1? No. C10 is row 1, col 0.
                           # My index convention for C_coeffs: [a][d] where a is row, d is col.
                           # C10 -> a=1, d=0.
                           # So P1: C10=1, C11=-1.
                           # [[0,0], [1,0]] for P1?
                           # Let's check P1 again. C10 = P1 + P3. So P1 coeff for C10 is 1.
                           # C11 = P0 - P1 + ... So P1 coeff for C11 is -1.
                           # So P1: [[0, 0], [1, -1]]
        [[0, 1], [1, 0]],  # P2: C01=1, C11=1.
                           # C01 -> a=0, d=1.
                           # C11 -> a=1, d=1.
                           # So P2: [[0, 1], [0, 1]]
        [[1, 0], [1, 0]],  # P3: C00=1, C10=1.
                           # C00 -> a=0, d=0.
                           # C10 -> a=1, d=0.
                           # So P3: [[1, 0], [1, 0]]
        [[-1, 1], [0, 0]], # P4: C00=-1, C01=1.
                           # So P4: [[-1, 1], [0, 0]]
        [[0, 0], [0, 1]],  # P5: C11=1.
                           # So P5: [[0, 0], [0, 1]]
        [[1, 0], [0, 0]],  # P6: C00=1.
                           # So P6: [[1, 0], [0, 0]]
    ]
    
    # Final Check on C_coeffs:
    # P0: [[1,0],[1,0]] -> C00+=1, C11+=1
    # P1: [[0,0],[1,-1]] -> C10+=1, C11+=-1
    # P2: [[0,1],[0,1]] -> C01+=1, C11+=1
    # P3: [[1,0],[1,0]] -> C00+=1, C10+=1
    # P4: [[-1,1],[0,0]] -> C00+=-1, C01+=1
    # P5: [[0,0],[0,1]] -> C11+=1
    # P6: [[1,0],[0,0]] -> C00+=1
    
    # Summing for C00: 1 (P0) + 1 (P3) - 1 (P4) + 1 (P6) = 2?
    # Strassen C00 = P0 + P3 - P4 + P6.
    # If inputs are identity matrices, P0=1, P3=0, P4=0, P6=0. C00=1.
    # My formula gives 1.
    # Wait, P0 coeff for C00 is 1. P3 is 1. P4 is -1. P6 is 1.
    # Sum = 1+1-1+1 = 2?
    # No, the coefficients are added.
    # C00 = 1*P0 + 1*P3 - 1*P4 + 1*P6.
    # In my matrix C_coeffs, I store the coefficient of P_r in C_{a,d}.
    # So for C00 (a=0, d=0):
    # P0: 1
    # P3: 1
    # P4: -1
    # P6: 1
    # This matches.
    
    # Let's re-verify P1.
    # C10 = P1 + P3.
    # C11 = P0 - P1 + P2 + P5.
    # So P1 coeff for C10 is 1.
    # P1 coeff for C11 is -1.
    # My C_coeffs[1] is [[0,0],[1,-1]].
    # a=1, d=0 is 1. Correct.
    # a=1, d=1 is -1. Correct.
    
    # P2.
    # C01 = P2 + P4.
    # C11 = ... + P2 ...
    # So P2 coeff for C01 is 1.
    # P2 coeff for C11 is 1.
    # My C_coeffs[2] is [[0,1],[0,1]].
    # a=0, d=1 is 1. Correct.
    # a=1, d=1 is 1. Correct.
    
    # P3.
    # C00 = ... + P3 ...
    # C10 = P1 + P3.
    # So P3 coeff for C00 is 1.
    # P3 coeff for C10 is 1.
    # My C_coeffs[3] is [[1,0],[1,0]].
    # a=0, d=0 is 1. Correct.
    # a=1, d=0 is 1. Correct.
    
    # P4.
    # C00 = ... - P4 ...
    # C01 = P2 + P4.
    # So P4 coeff for C00 is -1.
    # P4 coeff for C01 is 1.
    # My C_coeffs[4] is [[-1,1],[0,0]].
    # a=0, d=0 is -1. Correct.
    # a=0, d=1 is 1. Correct.
    
    # P5.
    # C11 = ... + P5 ...
    # My C_coeffs[5] is [[0,0],[0,1]].
    # a=1, d=1 is 1. Correct.
    
    # P6.
    # C00 = ... + P6 ...
    # My C_coeffs[6] is [[1,0],[0,0]].
    # a=0, d=0 is 1. Correct.
    
    # Blocks definition
    # block_id: 0, 1, 2, 3
    # A_col_offset: 0, 0, 2, 2
    # B_row_offset: 0, 0, 2, 2
    # B_col_offset: 0, 2, 0, 2
    
    blocks = [
        (0, 0, 0), # A_off, B_row_off, B_col_off
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    for b_idx, (a_off, br_off, bc_off) in enumerate(blocks):
        for p_idx in range(7):
            col = b_idx * 7 + p_idx
            
            # Update U
            for a in range(2):
                for b in range(2):
                    val = A_coeffs[p_idx][a][b]
                    if val != 0:
                        # Global A row: i=a, j=b+a_off
                        # U index: i*m + j
                        u_row = a * m + (b + a_off)
                        U[u_row][col] += val
                        
            # Update V
            for c in range(2):
                for d in range(2):
                    val = B_coeffs[p_idx][c][d]
                    if val != 0:
                        # Global B row: j=c+br_off, k=d+bc_off
                        # V index: j*p + k
                        v_row = (c + br_off) * p + (d + bc_off)
                        V[v_row][col] += val
                        
            # Update W
            for a in range(2):
                for d in range(2):
                    val = C_coeffs[p_idx][a][d]
                    if val != 0:
                        # Global C row: i=a, k=d+bc_off
                        # W index: k*n + i
                        w_row = (d + bc_off) * n + a
                        W[w_row][col] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
