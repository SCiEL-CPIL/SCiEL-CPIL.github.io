import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if n != 4 or m != 4 or p != 4:
        # Fallback to schoolbook for non-4x4x4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2 factors
    # U2, V2, W2 are 4x7 matrices
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V2 = [
        [1, 1, 0, -1, 1, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=1, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=0, i=1) -> Wait. W[k*n+i]. n=2. W[0*2+0]=C00, W[1*2+0]=C10? 
                                 # Problem: W[k*n+i]. 
                                 # For 2x2:
                                 # k=0, i=0 -> W[0] = C00
                                 # k=1, i=0 -> W[2] = C10? No.
                                 # C is 2x2. C[row, col].
                                 # Usually C[i, j] = sum A[i, k] B[k, j].
                                 # Here W index is k*n+i.
                                 # If n=2, W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]?
                                 # Let's check the schoolbook code:
                                 # W[k*n+i][column] = 1.
                                 # i is row of A/C. k is col of B/C.
                                 # So W index is k*n + i.
                                 # For n=2:
                                 # k=0, i=0 -> 0. C[0,0]
                                 # k=0, i=1 -> 1. C[1,0]
                                 # k=1, i=0 -> 2. C[0,1]
                                 # k=1, i=1 -> 3. C[1,1]
                                 # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11.
                                 # My W2 above assumed row-major C00, C01, C10, C11.
                                 # I need to permute W2 rows to match W[k*n+i].
        
        # Let's re-verify W2 indices.
        # C00: k=0, i=0 -> idx 0.
        # C10: k=0, i=1 -> idx 1.
        # C01: k=1, i=0 -> idx 2.
        # C11: k=1, i=1 -> idx 3.
        
        # My previous W2 rows:
        # Row 0: C00 coeffs.
        # Row 1: C01 coeffs.
        # Row 2: C10 coeffs.
        # Row 3: C11 coeffs.
        
        # So I need to reorder rows of W2 to:
        # New Row 0: Old Row 0 (C00)
        # New Row 1: Old Row 2 (C10)
        # New Row 2: Old Row 1 (C01)
        # New Row 3: Old Row 3 (C11)
        
        [1, 0, 0, 1, -1, 0, 1],  # C00 (idx 0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (idx 1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (idx 2)
        [1, 0, -1, 0, 0, 1, 1]   # C11 (idx 3)
    ]
    
    # Verify W2 again with new ordering
    # M1: C00+1, C10+0, C01+0, C11+1. 
    # My W2 col 0: 1, 0, 0, 1. Correct.
    # M4: C00+1, C10+1, C01+0, C11+0.
    # My W2 col 3: 1, 1, 0, 0. Correct.
    # M5: C00-1, C10+0, C01+1, C11+0.
    # My W2 col 4: -1, 0, 1, 0. Correct.
    # M7: C00+1, C10+0, C01+0, C11+1.
    # My W2 col 6: 1, 0, 0, 1. Correct.
    # M3: C00+0, C10+0, C01+1, C11-1.
    # My W2 col 2: 0, 0, 1, -1. Correct.
    # M2: C00+0, C10+1, C01+0, C11+0.
    # My W2 col 1: 0, 1, 0, 0. Correct.
    # M6: C00+0, C10+0, C01+0, C11+1.
    # My W2 col 5: 0, 0, 0, 1. Correct.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i in range(4):
                for j in range(4):
                    i_out = i // 2
                    i_in = i % 2
                    j_out = j // 2
                    j_in = j % 2
                    u_val = U2[i_out * 2 + j_out][c1] * U2[i_in * 2 + j_in][c2]
                    U[i * 4 + j][col] = u_val
                    
                    v_val = V2[j_out * 2 + i_out][c1] * V2[j_in * 2 + i_in][c2] 
                    # Wait. V index is j*p+k. 
                    # In 2x2, V index is j*2+k.
                    # j_out, k_out correspond to outer block indices.
                    # j_in, k_in correspond to inner block indices.
                    # For V (B matrix), the "row" is j, "col" is k.
                    # In the Kronecker expansion:
                    # B_{j, k} = B_{j_out, k_out} (block) * B_{j_in, k_in} (local)?
                    # No. B is 4x4.
                    # B_{j, k} where j, k in 0..3.
                    # j = 2 j_out + j_in.
                    # k = 2 k_out + k_in.
                    # The Strassen product M_{c1, c2} involves B blocks.
                    # The coefficient of B_{j_out, k_out} in M_{c1} is V2[2 j_out + k_out][c1].
                    # The coefficient of B_{j_in, k_in} in M_{c2} is V2[2 j_in + k_in][c2].
                    # So V4[j*4+k, col] = V2[2 j_out + k_out][c1] * V2[2 j_in + k_in][c2].
                    # But k is not in the loop for U.
                    # We need to iterate k as well for V and W.
                    
            # It is more efficient to loop over all indices for U, V, W separately or together.
            # Let's restructure the inner loop to fill U, V, W for all indices.
            
    # Re-implementing the fill loop correctly
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    i_out = i // 2
                    i_in = i % 2
                    j_out = j // 2
                    j_in = j % 2
                    U[i * 4 + j][col] = U2[i_out * 2 + j_out][c1] * U2[i_in * 2 + j_in][c2]
                    
            # Fill V
            for j in range(4):
                for k in range(4):
                    j_out = j // 2
                    j_in = j % 2
                    k_out = k // 2
                    k_in = k % 2
                    V[j * 4 + k][col] = V2[j_out * 2 + k_out][c1] * V2[j_in * 2 + k_in][c2]
                    
            # Fill W
            for k in range(4):
                for i in range(4):
                    k_out = k // 2
                    k_in = k % 2
                    i_out = i // 2
                    i_in = i % 2
                    W[k * 4 + i][col] = W2[k_out * 2 + i_out][c1] * W2[k_in * 2 + i_in][c2]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
