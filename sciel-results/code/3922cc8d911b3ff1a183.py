import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen steps = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 blocks
    # Block 1: A_L (i, j=0,1), B_LL (j=0,1, k=0,1) -> C(i, k=0,1)
    # Block 2: A_R (i, j=2,3), B_RL (j=2,3, k=0,1) -> C(i, k=0,1)
    # Block 3: A_L (i, j=0,1), B_LR (j=0,1, k=2,3) -> C(i, k=2,3)
    # Block 4: A_R (i, j=2,3), B_RR (j=2,3, k=2,3) -> C(i, k=2,3)
    
    blocks = [
        # (i_offset, j_offset_A, j_offset_B, k_offset)
        (0, 0, 0, 0), # Block 1
        (0, 2, 2, 0), # Block 2
        (0, 0, 0, 2), # Block 3
        (0, 2, 2, 2), # Block 4
    ]
    
    # Strassen multipliers and coefficients
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each block, we add 7 columns
    for i_off, j_off_A, j_off_B, k_off in blocks:
        # Process 7 Strassen steps
        for step in range(7):
            # Determine U coefficients (from A)
            # U[i*m + j, col]
            # Determine V coefficients (from B)
            # V[j*p + k, col]
            # Determine W coefficients (to C)
            # W[k*n + i, col]
            
            # X is 2x2 submatrix of A, Y is 2x2 submatrix of B
            # X_ii, X_ij correspond to A[i_off + i, j_off_A + j]
            # Y_ij, Y_jk correspond to B[j_off_B + i, k_off + k] ? 
            # Wait, B is 4x4.
            # In Block 1: B_LL is rows 0,1 and cols 0,1 of B.
            # So B[j_off_B + i, k_off + k] is correct for local indices i,k in 0..1.
            
            # Let's define the local indices for X and Y
            # X is 2x2. Local indices (li, lj) in {0,1}^2.
            # Global A index: (i_off + li, j_off_A + lj)
            # Y is 2x2. Local indices (li, lk) in {0,1}^2.
            # Global B index: (j_off_B + li, k_off + lk)
            
            # Strassen Step 1: M1 = (X00 + X11) * (Y00 + Y11)
            # U part: X00, X11
            # V part: Y00, Y11
            # W part: contributes to Z00, Z11
            
            # We will build the 7 steps explicitly
            
            if step == 0: # M1
                # U: A[0,0] + A[1,1]
                U[(i_off+0)*m + (j_off_A+0)][col] = 1
                U[(i_off+1)*m + (j_off_A+1)][col] = 1
                # V: B[0,0] + B[1,1]
                V[(j_off_B+0)*p + (k_off+0)][col] = 1
                V[(j_off_B+1)*p + (k_off+1)][col] = 1
                # W: Z00 += M1, Z11 += M1
                W[(k_off+0)*n + (i_off+0)][col] = 1
                W[(k_off+1)*n + (i_off+1)][col] = 1
                
            elif step == 1: # M2
                # M2 = (X10 + X11) * Y00
                U[(i_off+1)*m + (j_off_A+0)][col] = 1
                U[(i_off+1)*m + (j_off_A+1)][col] = 1
                V[(j_off_B+0)*p + (k_off+0)][col] = 1
                # W: Z10 += M2
                W[(k_off+0)*n + (i_off+1)][col] = 1
                
            elif step == 2: # M3
                # M3 = X00 * (Y01 - Y11)
                U[(i_off+0)*m + (j_off_A+0)][col] = 1
                V[(j_off_B+0)*p + (k_off+1)][col] = 1
                V[(j_off_B+1)*p + (k_off+1)][col] = -1
                # W: Z01 += M3, Z11 += M3
                W[(k_off+1)*n + (i_off+0)][col] = 1
                W[(k_off+1)*n + (i_off+1)][col] = 1
                
            elif step == 3: # M4
                # M4 = X11 * (Y10 - Y00)
                U[(i_off+1)*m + (j_off_A+1)][col] = 1
                V[(j_off_B+1)*p + (k_off+0)][col] = 1
                V[(j_off_B+0)*p + (k_off+0)][col] = -1
                # W: Z00 += M4, Z10 += M4
                W[(k_off+0)*n + (i_off+0)][col] = 1
                W[(k_off+0)*n + (i_off+1)][col] = 1
                
            elif step == 4: # M5
                # M5 = (X00 + X01) * Y11
                U[(i_off+0)*m + (j_off_A+0)][col] = 1
                U[(i_off+0)*m + (j_off_A+1)][col] = 1
                V[(j_off_B+1)*p + (k_off+1)][col] = 1
                # W: Z00 -= M5, Z01 += M5
                W[(k_off+0)*n + (i_off+0)][col] = -1
                W[(k_off+1)*n + (i_off+0)][col] = 1
                
            elif step == 5: # M6
                # M6 = (X10 - X00) * (Y00 + Y01)
                U[(i_off+1)*m + (j_off_A+0)][col] = 1
                U[(i_off+0)*m + (j_off_A+0)][col] = -1
                V[(j_off_B+0)*p + (k_off+0)][col] = 1
                V[(j_off_B+0)*p + (k_off+1)][col] = 1
                # W: Z11 += M6
                W[(k_off+1)*n + (i_off+1)][col] = 1
                
            elif step == 6: # M7
                # M7 = (X01 - X11) * (Y10 + Y11)
                U[(i_off+0)*m + (j_off_A+1)][col] = 1
                U[(i_off+1)*m + (j_off_A+1)][col] = -1
                V[(j_off_B+1)*p + (k_off+0)][col] = 1
                V[(j_off_B+1)*p + (k_off+1)][col] = 1
                # W: Z00 += M7
                W[(k_off+0)*n + (i_off+0)][col] = 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
