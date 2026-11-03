"""Rewritten program for the suite task; improved rank using Strassen for 2x2x8 block."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x3x8 tensor into:
    # 1. 2x2x8 block (A[0:2, 0:2] * B[0:2, :])
    # 2. 2x1x8 block (A[0:2, 2]   * B[2, :])
    # 3. 1x2x8 block (A[2, 0:2]   * B[0:2, :])
    # 4. 1x1x8 block (A[2, 2]     * B[2, :])
    
    # Rank contributions:
    # 1. R(2x2x8) <= 4 * R(2x2x2) = 4 * 7 = 28
    # 2. R(2x1x8) = 2 * 8 = 16
    # 3. R(1x2x8) = 1 * 2 * 8 = 16
    # 4. R(1x1x8) = 1 * 8 = 8
    # Total Rank = 28 + 16 + 16 + 8 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Block 1: 2x2x8 using Strassen (4 blocks of 2x2x2) ---
    # Strassen coefficients for 2x2x2
    # U/V/W indices for 2x2x2 sub-problem
    # A is 2x2, B is 2x2
    # U indices: i*m + j -> i*2 + j
    # V indices: j*p + k -> j*2 + k
    # W indices: k*n + i -> k*2 + i
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 1, 1, 0], # M2? No, let's use standard form
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    # It's easier to generate the 7 terms explicitly for each of the 4 blocks.
    
    def add_strassen_block(a_row_start, b_col_start, a_col_end, b_col_end, current_col):
        # a_row_start: start row of A in 2x2 block (0 or 1 relative to block)
        # b_col_start: start col of B in 2x2 block (0, 2, 4, 6 relative to full B)
        # We are solving A[0:2, 0:2] * B[0:2, b_col_start:b_col_start+2]
        
        # The 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We map these to the global U, V, W matrices.
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {0,1} (relative to block)
        
        # U[i*2+j, col]
        # V[j*2+k, col]
        # W[k*2+i, col]
        
        # Note: The B columns are offset by b_col_start in the global B.
        # But V is defined on the full m*p space.
        # V index is j*p + k_global.
        # Here j is 0 or 1 (relative to A block).
        # k_global is b_col_start + k_rel.
        
        for r in range(7):
            c = current_col + r
            # Clear U, V, W for this column (already 0)
            
            # Term r=0: M1
            if r == 0:
                # (A00+A11)
                U[0*2+0][c] = 1
                U[1*2+1][c] = 1
                # (B00+B11)
                V[0*p + (b_col_start + 0)][c] = 1
                V[1*p + (b_col_start + 1)][c] = 1
                # Output: C00 += M1, C01 += M1, C10 += M1, C11 += M1?
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # W for C00 (k=0,i=0): W[0*2+0]
                # W for C01 (k=1,i=0): W[1*2+0]
                # W for C10 (k=0,i=1): W[0*2+1]
                # W for C11 (k=1,i=1): W[1*2+1]
                
                W[0*2+0][c] = 1
                W[1*2+0][c] = 1
                W[0*2+1][c] = 1
                W[1*2+1][c] = 1
            elif r == 1: # M2
                # (A10+A11)
                U[1*2+0][c] = 1
                U[1*2+1][c] = 1
                # B00
                V[0*p + (b_col_start + 0)][c] = 1
                # C10 += M2, C11 -= M2
                W[0*2+1][c] = 1
                W[1*2+1][c] = -1
            elif r == 2: # M3
                # A00
                U[0*2+0][c] = 1
                # (B10-B11)
                V[1*p + (b_col_start + 0)][c] = 1
                V[1*p + (b_col_start + 1)][c] = -1
                # C01 += M3, C11 += M3
                W[1*2+0][c] = 1
                W[1*2+1][c] = 1
            elif r == 3: # M4
                # A11
                U[1*2+1][c] = 1
                # (B01-B00)
                V[0*p + (b_col_start + 1)][c] = 1
                V[0*p + (b_col_start + 0)][c] = -1
                # C00 += M4, C10 += M4
                W[0*2+0][c] = 1
                W[0*2+1][c] = 1
            elif r == 4: # M5
                # (A00+A01)
                U[0*2+0][c] = 1
                U[0*2+1][c] = 1
                # B11
                V[1*p + (b_col_start + 1)][c] = 1
                # C00 -= M5, C01 += M5
                W[0*2+0][c] = -1
                W[1*2+0][c] = 1
            elif r == 5: # M6
                # (A10-A00)
                U[1*2+0][c] = 1
                U[0*2+0][c] = -1
                # (B00+B01)
                V[0*p + (b_col_start + 0)][c] = 1
                V[0*p + (b_col_start + 1)][c] = 1
                # C11 += M6
                W[1*2+1][c] = 1
            elif r == 6: # M7
                # (A01-A11)
                U[0*2+1][c] = 1
                U[1*2+1][c] = -1
                # (B10+B11)
                V[1*p + (b_col_start + 0)][c] = 1
                V[1*p + (b_col_start + 1)][c] = 1
                # C00 += M7
                W[0*2+0][c] = 1
        
        return current_col + 7

    # Apply Strassen for 4 blocks of 2x2x2
    col = add_strassen_block(0, 0, 2, 2, col)
    col = add_strassen_block(0, 2, 2, 4, col)
    col = add_strassen_block(0, 4, 2, 6, col)
    col = add_strassen_block(0, 6, 2, 8, col)
    
    # --- Block 2: 2x1x8 (A[0:2, 2] * B[2, :]) ---
    # Rank 16. Schoolbook for this sub-block.
    for i in range(2):
        for k in range(8):
            c = col
            U[i * m + 2][c] = 1  # A[i, 2]
            V[2 * p + k][c] = 1  # B[2, k]
            W[k * n + i][c] = 1  # C[i, k]
            col += 1
            
    # --- Block 3: 1x2x8 (A[2, 0:2] * B[0:2, :]) ---
    # Rank 16. Schoolbook.
    for j in range(2):
        for k in range(8):
            c = col
            U[2 * m + j][c] = 1  # A[2, j]
            V[j * p + k][c] = 1  # B[j, k]
            W[k * n + 2][c] = 1  # C[2, k]
            col += 1

    # --- Block 4: 1x1x8 (A[2, 2] * B[2, :]) ---
    # Rank 8. Schoolbook.
    for k in range(8):
        c = col
        U[2 * m + 2][c] = 1  # A[2, 2]
        V[2 * p + k][c] = 1  # B[2, k]
        W[k * n + 2][c] = 1  # C[2, k]
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
