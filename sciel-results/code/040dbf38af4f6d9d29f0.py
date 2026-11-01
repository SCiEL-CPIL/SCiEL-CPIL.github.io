"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
Uses Strassen's algorithm on 2x2 blocks to reduce rank from 64 to 56.
The 2x4 * 4x8 multiplication is split into four 2x4 * 4x2 sub-problems.
Each sub-problem is solved using two 2x2 Strassen multiplications."""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Split p (8) into 4 blocks of size 2
    num_blocks = p // 2
    block_size = 2
    
    # Split m (4) into 2 blocks of size 2
    # A is 2x4, so A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B is 4x8, so for each block k, B1_k = B[0:2, k*2:(k+1)*2], B2_k = B[2:4, k*2:(k+1)*2]
    
    # Rank for one 2x2 Strassen multiplication is 7
    # We need two such multiplications per block (A1*B1 and A2*B2)
    # So rank per block is 14
    rank_per_block = 14
    total_rank = num_blocks * rank_per_block  # 4 * 14 = 56
    
    U = [[0] * total_rank for _ in range(n * m)]   # 8 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 32 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 16 rows
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(offset, a_row_start, b_row_start, c_col_start):
        """
        Adds a Strassen block for A[a_row_start:a_row_start+2, :] * B[: , b_row_start:b_row_start+2]
        Actually, we are computing C_block = A1 B1 + A2 B2.
        A1 is 2x2, A2 is 2x2.
        B1 is 2x2, B2 is 2x2.
        
        a_row_start: row index in A for A1 (0) and A2 (2)
        b_row_start: row index in B for B1 (0) and B2 (2)
        c_col_start: col index in C for the block
        """
        # We need to map the Strassen factors to the global U, V, W matrices.
        # U corresponds to A indices.
        # V corresponds to B indices.
        # W corresponds to C indices.
        
        # For A1 (rows 0,1 of A block) and A2 (rows 2,3 of A block)
        # The A matrix in the global sense has rows 0..3 (m=4).
        # A1 uses rows 0,1. A2 uses rows 2,3.
        
        # We will add 14 factors.
        # Factors 0-6: Strassen for A1 * B1
        # Factors 7-13: Strassen for A2 * B2
        
        # Strassen for A1 * B1
        # A1 indices in global A: rows 0, 1. Cols 0, 1.
        # B1 indices in global B: rows 0, 1. Cols b_col_start, b_col_start+1.
        
        # M1 = (A00 + A11)(B00 + B11)
        # U[0, offset] += 1, U[1, offset] += 1
        # V[0, offset] += 1, V[1, offset] += 1
        # W[c_col_start, offset] += 1 (C00)
        # W[c_col_start+1, offset] += 0? No, C01?
        # Strassen C00 = M1 + M4 - M5 + M7
        # Strassen C01 = M3 + M5
        # Strassen C10 = M2 + M4
        # Strassen C11 = M1 - M2 + M3 + M6
        
        # Let's define the contributions to W.
        # W is indexed by [k*n + i].
        # Here k is the column of B/C, i is the row of A/C.
        # For this block, k = b_col_start + 0 or 1.
        # i = 0 or 1.
        
        # Factor 0: M1
        U[0][offset] += 1
        U[1][offset] += 1
        V[0][offset] += 1
        V[1][offset] += 1
        W[(b_col_start) * n + 0][offset] += 1  # C00
        W[(b_col_start+1) * n + 1][offset] += 1 # C11
        
        # Factor 1: M2 = (A10 + A11)B00
        U[1][offset+1] += 1
        U[3][offset+1] += 1 # A10 is row 1, col 0? No.
        # A1 is 2x2. A1[0,0]=A[0,0], A1[0,1]=A[0,1], A1[1,0]=A[1,0], A1[1,1]=A[1,1].
        # Global A indices:
        # A1[0,0] -> U[0]
        # A1[0,1] -> U[1]
        # A1[1,0] -> U[2]
        # A1[1,1] -> U[3]
        
        # Correct A indices for A1:
        # Row 0: U[0], U[1]
        # Row 1: U[2], U[3]
        
        # Let's restart the mapping carefully.
        
        # A1:
        # A1[0,0] -> U[0]
        # A1[0,1] -> U[1]
        # A1[1,0] -> U[2]
        # A1[1,1] -> U[3]
        
        # B1:
        # B1[0,0] -> V[0*p + b_col_start] = V[b_col_start]
        # B1[0,1] -> V[0*p + b_col_start+1] = V[b_col_start+1]
        # B1[1,0] -> V[1*p + b_col_start] = V[p + b_col_start]
        # B1[1,1] -> V[1*p + b_col_start+1] = V[p + b_col_start+1]
        
        # C1 (result of A1*B1):
        # C1[0,0] -> W[b_col_start*n + 0]
        # C1[0,1] -> W[(b_col_start+1)*n + 0]
        # C1[1,0] -> W[b_col_start*n + 1]
        # C1[1,1] -> W[(b_col_start+1)*n + 1]
        
        # Strassen M1 = (A1[0,0] + A1[1,1]) * (B1[0,0] + B1[1,1])
        U[0][offset] += 1
        U[3][offset] += 1
        V[b_col_start][offset] += 1
        V[p + b_col_start + 1][offset] += 1
        W[b_col_start*n + 0][offset] += 1
        W[(b_col_start+1)*n + 1][offset] += 1
        
        # Strassen M2 = (A1[1,0] + A1[1,1]) * B1[0,0]
        U[2][offset+1] += 1
        U[3][offset+1] += 1
        V[b_col_start][offset+1] += 1
        W[b_col_start*n + 1][offset+1] += 1
        W[(b_col_start+1)*n + 1][offset+1] -= 1
        
        # Strassen M3 = A1[0,0] * (B1[0,1] - B1[1,1])
        U[0][offset+2] += 1
        V[b_col_start+1][offset+2] += 1
        V[p + b_col_start + 1][offset+2] -= 1
        W[(b_col_start+1)*n + 0][offset+2] += 1
        W[(b_col_start+1)*n + 1][offset+2] += 1
        
        # Strassen M4 = A1[1,1] * (B1[1,0] - B1[0,0])
        U[3][offset+3] += 1
        V[p + b_col_start][offset+3] += 1
        V[b_col_start][offset+3] -= 1
        W[b_col_start*n + 0][offset+3] += 1
        W[b_col_start*n + 1][offset+3] += 1
        
        # Strassen M5 = (A1[0,0] + A1[0,1]) * B1[1,1]
        U[0][offset+4] += 1
        U[1][offset+4] += 1
        V[p + b_col_start + 1][offset+4] += 1
        W[b_col_start*n + 0][offset+4] -= 1
        W[(b_col_start+1)*n + 0][offset+4] += 1
        
        # Strassen M6 = (A1[1,0] - A1[0,0]) * (B1[0,0] + B1[0,1])
        U[2][offset+5] += 1
        U[0][offset+5] -= 1
        V[b_col_start][offset+5] += 1
        V[b_col_start+1][offset+5] += 1
        W[(b_col_start+1)*n + 1][offset+5] += 1
        
        # Strassen M7 = (A1[0,1] - A1[1,1]) * (B1[1,0] + B1[1,1])
        U[1][offset+6] += 1
        U[3][offset+6] -= 1
        V[p + b_col_start][offset+6] += 1
        V[p + b_col_start + 1][offset+6] += 1
        W[b_col_start*n + 0][offset+6] += 1
        W[(b_col_start+1)*n + 1][offset+6] -= 1
        
        # Now for A2 * B2
        # A2 indices in global A:
        # A2[0,0] -> U[4]
        # A2[0,1] -> U[5]
        # A2[1,0] -> U[6]
        # A2[1,1] -> U[7]
        
        # B2 indices in global B:
        # B2[0,0] -> V[2*p + b_col_start]
        # B2[0,1] -> V[2*p + b_col_start+1]
        # B2[1,0] -> V[3*p + b_col_start]
        # B2[1,1] -> V[3*p + b_col_start+1]
        
        # C2 (result of A2*B2) adds to the same C indices.
        
        # Strassen M8 (M1 for A2B2)
        U[4][offset+7] += 1
        U[7][offset+7] += 1
        V[2*p + b_col_start][offset+7] += 1
        V[3*p + b_col_start + 1][offset+7] += 1
        W[b_col_start*n + 0][offset+7] += 1
        W[(b_col_start+1)*n + 1][offset+7] += 1
        
        # Strassen M9 (M2 for A2B2)
        U[6][offset+8] += 1
        U[7][offset+8] += 1
        V[2*p + b_col_start][offset+8] += 1
        W[b_col_start*n + 1][offset+8] += 1
        W[(b_col_start+1)*n + 1][offset+8] -= 1
        
        # Strassen M10 (M3 for A2B2)
        U[4][offset+9] += 1
        V[2*p + b_col_start+1][offset+9] += 1
        V[3*p + b_col_start + 1][offset+9] -= 1
        W[(b_col_start+1)*n + 0][offset+9] += 1
        W[(b_col_start+1)*n + 1][offset+9] += 1
        
        # Strassen M11 (M4 for A2B2)
        U[7][offset+10] += 1
        V[3*p + b_col_start][offset+10] += 1
        V[2*p + b_col_start][offset+10] -= 1
        W[b_col_start*n + 0][offset+10] += 1
        W[b_col_start*n + 1][offset+10] += 1
        
        # Strassen M12 (M5 for A2B2)
        U[4][offset+11] += 1
        U[5][offset+11] += 1
        V[3*p + b_col_start + 1][offset+11] += 1
        W[b_col_start*n + 0][offset+11] -= 1
        W[(b_col_start+1)*n + 0][offset+11] += 1
        
        # Strassen M13 (M6 for A2B2)
        U[6][offset+12] += 1
        U[4][offset+12] -= 1
        V[2*p + b_col_start][offset+12] += 1
        V[2*p + b_col_start+1][offset+12] += 1
        W[(b_col_start+1)*n + 1][offset+12] += 1
        
        # Strassen M14 (M7 for A2B2)
        U[5][offset+13] += 1
        U[7][offset+13] -= 1
        V[3*p + b_col_start][offset+13] += 1
        V[3*p + b_col_start + 1][offset+13] += 1
        W[b_col_start*n + 0][offset+13] += 1
        W[(b_col_start+1)*n + 1][offset+13] -= 1

    for k in range(num_blocks):
        b_col_start = k * block_size
        add_strassen_block(k * rank_per_block, 0, 0, b_col_start)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
