"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the Strassen-based rank-56 decomposition of the 2x4 by 4x8 matrix multiplication tensor.
    
    The 4x8 matrix B is split into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
    Each 2x4 by 4x4 multiplication is decomposed into four 2x2 block multiplications.
    Each 2x2 multiplication is decomposed using Strassen's algorithm (rank 7 instead of 8).
    Total rank: 4 blocks * 7 = 28 per 2x4x4, and 28 * 2 = 56 for the full 2x4x8.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # C11 = M1 - M2
    # C12 = M3 + M4
    # C21 = M5 + M2
    # C22 = M1 - M5
    # where:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # We need to construct U, V, W matrices.
    # U has shape (n*m, rank) = (8, 56)
    # V has shape (m*p, rank) = (32, 56)
    # W has shape (p*n, rank) = (16, 56)
    
    # Index conventions:
    # U[i*m + j, c] where i in [0,2), j in [0,4)
    # V[j*p + k, c] where j in [0,4), k in [0,8)
    # W[k*n + i, c] where k in [0,8), i in [0,2)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each of the two 4x4 blocks of B (left and right)
    for block_idx in range(2):
        # block_idx 0: cols 0-3 of B, contributes to cols 0-3 of C
        # block_idx 1: cols 4-7 of B, contributes to cols 4-7 of C
        k_offset = block_idx * 4  # offset in the k (column of B / row of C) dimension
        
        # For each of the four 2x2 blocks in the 2x4 x 4x4 multiplication
        # A is 2x4, split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
        # B_block is 4x4, split into four 2x2 blocks
        # C_block is 2x4, split into C_left (2x2, cols 0-1) and C_right (2x2, cols 2-3)
        
        # The four 2x2 block multiplications:
        # C_left = A_left * B_top_left + A_right * B_bottom_left
        # C_right = A_left * B_top_right + A_right * B_bottom_right
        
        # Block positions in the 2x4 x 4x4:
        # A_left: rows 0-1, cols 0-1 of A (i.e., j=0,1)
        # A_right: rows 0-1, cols 2-3 of A (i.e., j=2,3)
        # B_top_left: rows 0-1, cols 0-1 of B_block (i.e., j=0,1, k=0,1 relative)
        # B_top_right: rows 0-1, cols 2-3 of B_block (i.e., j=0,1, k=2,3 relative)
        # B_bottom_left: rows 2-3, cols 0-1 of B_block (i.e., j=2,3, k=0,1 relative)
        # B_bottom_right: rows 2-3, cols 2-3 of B_block (i.e., j=2,3, k=2,3 relative)
        
        # For each pair (A_block, B_block) that contributes to a C_block:
        # There are 4 such pairs:
        # 1. A_left * B_top_left -> contributes to C_left
        # 2. A_right * B_bottom_left -> contributes to C_left
        # 3. A_left * B_top_right -> contributes to C_right
        # 4. A_right * B_bottom_right -> contributes to C_right
        
        # Define the 4 blocks:
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset, c_row_offset)
        # a_row is always 0-1 (since A is 2 rows)
        # a_col_offset: 0 for left, 2 for right
        # b_row_offset: 0 for top, 2 for bottom
        # b_col_offset: 0 for left, 2 for right
        # c_col_offset: 0 for left result, 2 for right result
        # c_row_offset: always 0-1 (since C is 2 rows)
        
        blocks = [
            # (a_col_off, b_row_off, b_col_off, c_col_off)
            (0, 0, 0, 0),  # A_left * B_top_left -> C_left
            (2, 2, 0, 0),  # A_right * B_bottom_left -> C_left
            (0, 0, 2, 2),  # A_left * B_top_right -> C_right
            (2, 2, 2, 2),  # A_right * B_bottom_right -> C_right
        ]
        
        for a_co, b_ro, b_co, c_co in blocks:
            # Strassen's 7 products for this 2x2 block
            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B12-B22)
            # M4 = A22*(B21-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)
            
            # In our indexing:
            # A is 2x2 with elements A[0,0], A[0,1], A[1,0], A[1,1]
            #   A[0,0] = A[i=0, j=a_co+0], A[0,1] = A[i=0, j=a_co+1]
            #   A[1,0] = A[i=1, j=a_co+0], A[1,1] = A[i=1, j=a_co+1]
            # B is 2x2 with elements B[0,0], B[0,1], B[1,0], B[1,1]
            #   B[0,0] = B[j=b_ro+0, k=k_offset+b_co+0], etc.
            # C is 2x2 with elements C[0,0], C[0,1], C[1,0], C[1,1]
            #   C[0,0] = C[i=0, k=k_offset+c_co+0], etc.
            
            # M1 = (A00+A11)*(B00+B11)
            # U contributions: A[0, a_co+0] + A[1, a_co+0]... wait, let me re-index
            # A[0,0] = U index (0, a_co), A[0,1] = U index (0, a_co+1)
            # A[1,0] = U index (1, a_co), A[1,1] = U index (1, a_co+1)
            
            # M1: (A00 + A11) * (B00 + B11)
            # A00 = A[0, a_co], A11 = A[1, a_co+1]
            # B00 = B[b_ro, k_offset+b_co], B11 = B[b_ro+1, k_offset+b_co+1]
            # C contributions: C00 += M1, C01 -= M1 (no, let me use the standard Strassen)
            
            # Standard Strassen for C = A*B:
            # C[0,0] = M1 - M2
            # C[0,1] = M3 + M4
            # C[1,0] = M5 + M2
            # C[1,1] = M1 - M5
            
            # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            # M2 = (A[1,0] + A[1,1]) * B[0,0]
            # M3 = A[0,0] * (B[0,1] - B[1,1])
            # M4 = A[1,1] * (B[1,0] - B[0,0])
            # M5 = (A[0,0] + A[0,1]) * B[1,1]
            # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
            # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
            
            # Wait, I need to be careful. Let me use a cleaner formulation.
            # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
            # C = [[c00, c01], [c10, c11]]
            # c00 = a00*b00 + a01*b10
            # c01 = a00*b01 + a01*b11
            # c10 = a10*b00 + a11*b10
            # c11 = a10*b01 + a11*b11
            
            # Strassen:
            # m1 = (a00+a11)*(b00+b11)
            # m2 = (a10+a11)*b00
            # m3 = a00*(b01-b11)
            # m4 = a11*(b10-b00)
            # m5 = (a00+a01)*b11
            # m6 = (a10-a00)*(b00+b01)
            # m7 = (a01-a11)*(b10+b11)
            # c00 = m1 - m2
            # c01 = m3 + m4
            # c10 = m5 + m2
            # c11 = m1 - m5
            
            # Now map to our indices:
            # a00 = A[0, a_co], a01 = A[0, a_co+1]
            # a10 = A[1, a_co], a11 = A[1, a_co+1]
            # b00 = B[b_ro, k_offset+b_co], b01 = B[b_ro, k_offset+b_co+1]
            # b10 = B[b_ro+1, k_offset+b_co], b11 = B[b_ro+1, k_offset+b_co+1]
            # c00 = C[0, k_offset+c_co], c01 = C[0, k_offset+c_co+1]
            # c10 = C[1, k_offset+c_co], c11 = C[1, k_offset+c_co+1]
            
            # U indices:
            # U[0*m+0, col] for A[0,0], etc.
            # U[i*m + j, col]
            
            # m1 = (a00+a11)*(b00+b11)
            U[0*m + a_co][col] += 1      # a00
            U[1*m + a_co + 1][col] += 1  # a11
            V[b_ro*p + k_offset + b_co][col] += 1      # b00
            V[(b_ro+1)*p + k_offset + b_co + 1][col] += 1  # b11
            W[(k_offset + c_co)*n + 0][col] += 1       # c00
            W[(k_offset + c_co + 1)*n + 1][col] += 1   # c11
            col += 1
            
            # m2 = (a10+a11)*b00
            U[1*m + a_co][col] += 1      # a10
            U[1*m + a_co + 1][col] += 1  # a11
            V[b_ro*p + k_offset + b_co][col] += 1      # b00
            W[(k_offset + c_co)*n + 0][col] -= 1       # c00 -= m2
            W[(k_offset + c_co)*n + 1][col] += 1       # c10 += m2
            col += 1
            
            # m3 = a00*(b01-b11)
            U[0*m + a_co][col] += 1      # a00
            V[b_ro*p + k_offset + b_co + 1][col] += 1      # b01
            V[(b_ro+1)*p + k_offset + b_co + 1][col] -= 1  # -b11
            W[(k_offset + c_co + 1)*n + 0][col] += 1       # c01 += m3
            col += 1
            
            # m4 = a11*(b10-b00)
            U[1*m + a_co + 1][col] += 1  # a11
            V[(b_ro+1)*p + k_offset + b_co][col] += 1      # b10
            V[b_ro*p + k_offset + b_co][col] -= 1          # -b00
            W[(k_offset + c_co + 1)*n + 0][col] += 1       # c01 += m4
            col += 1
            
            # m5 = (a00+a01)*b11
            U[0*m + a_co][col] += 1      # a00
            U[0*m + a_co + 1][col] += 1  # a01
            V[(b_ro+1)*p + k_offset + b_co + 1][col] += 1  # b11
            W[(k_offset + c_co)*n + 1][col] += 1           # c10 += m5
            W[(k_offset + c_co + 1)*n + 1][col] -= 1       # c11 -= m5
            col += 1
            
            # m6 = (a10-a00)*(b00+b01)
            U[1*m + a_co][col] += 1      # a10
            U[0*m + a_co][col] -= 1      # -a00
            V[b_ro*p + k_offset + b_co][col] += 1      # b00
            V[b_ro*p + k_offset + b_co + 1][col] += 1  # b01
            # m6 doesn't directly contribute to C in this formulation... 
            # Wait, I think I made an error. Let me re-check Strassen.
            
            # Actually, the standard Strassen is:
            # p1 = (a00+a11)(b00+b11)
            # p2 = (a10+a11)b00
            # p3 = a00(b01-b11)
            # p4 = a11(b10-b00)
            # p5 = (a00+a01)b11
            # p6 = (a10-a00)(b00+b01)
            # p7 = (a01-a11)(b10+b11)
            # c00 = p1 - p2 + p4
            # c01 = p3 + p5
            # c10 = p2 + p6
            # c11 = p1 - p5 + p7
            
            # I need to redo this.
            
            # Let me restart the Strassen implementation more carefully.
            pass
    
    # I made errors in the Strassen coefficients. Let me redo this properly.
    # I'll rebuild from scratch with the correct Strassen formulas.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(2):
        k_offset = block_idx * 4
        
        blocks = [
            (0, 0, 0, 0),  # A_left * B_top_left -> C_left
            (2, 2, 0, 0),  # A_right * B_bottom_left -> C_left
            (0, 0, 2, 2),  # A_left * B_top_right -> C_right
            (2, 2, 2, 2),  # A_right * B_bottom_right -> C_right
        ]
        
        for a_co, b_ro, b_co, c_co in blocks:
            # Strassen's 7 products with correct C reconstruction:
            # p1 = (a00+a11)(b00+b11)
            # p2 = (a10+a11)b00
            # p3 = a00(b01-b11)
            # p4 = a11(b10-b00)
            # p5 = (a00+a01)b11
            # p6 = (a10-a00)(b00+b01)
            # p7 = (a01-a11)(b10+b11)
            # c00 = p1 - p2 + p4
            # c01 = p3 + p5
            # c10 = p2 + p6
            # c11 = p1 - p5 + p7
            
            # Indices:
            # a00 = A[0, a_co], a01 = A[0, a_co+1]
            # a10 = A[1, a_co], a11 = A[1, a_co+1]
            # b00 = B[b_ro, k_offset+b_co], b01 = B[b_ro, k_offset+b_co+1]
            # b10 = B[b_ro+1, k_offset+b_co], b11 = B[b_ro+1, k_offset+b_co+1]
            # c00 = C[0, k_offset+c_co], c01 = C[0, k_offset+c_co+1]
            # c10 = C[1, k_offset+c_co], c11 = C[1, k_offset+c_co+1]
            
            # p1 = (a00+a11)(b00+b11)
            U[0*m + a_co][col] += 1
            U[1*m + a_co + 1][col] += 1
            V[b_ro*p + k_offset + b_co][col] += 1
            V[(b_ro+1)*p + k_offset + b_co + 1][col] += 1
            W[(k_offset + c_co)*n + 0][col] += 1       # c00 += p1
            W[(k_offset + c_co + 1)*n + 1][col] += 1   # c11 += p1
            col += 1
            
            # p2 = (a10+a11)b00
            U[1*m + a_co][col] += 1
            U[1*m + a_co + 1][col] += 1
            V[b_ro*p + k_offset + b_co][col] += 1
            W[(k_offset + c_co)*n + 0][col] -= 1       # c00 -= p2
            W[(k_offset + c_co)*n + 1][col] += 1       # c10 += p2
            col += 1
            
            # p3 = a00(b01-b11)
            U[0*m + a_co][col] += 1
            V[b_ro*p + k_offset + b_co + 1][col] += 1
            V[(b_ro+1)*p + k_offset + b_co + 1][col] -= 1
            W[(k_offset + c_co + 1)*n + 0][col] += 1   # c01 += p3
            col += 1
            
            # p4 = a11(b10-b00)
            U[1*m + a_co + 1][col] += 1
            V[(b_ro+1)*p + k_offset + b_co][col] += 1
            V[b_ro*p + k_offset + b_co][col] -= 1
            W[(k_offset + c_co)*n + 0][col] += 1       # c00 += p4
            col += 1
            
            # p5 = (a00+a01)b11
            U[0*m + a_co][col] += 1
            U[0*m + a_co + 1][col] += 1
            V[(b_ro+1)*p + k_offset + b_co + 1][col] += 1
            W[(k_offset + c_co + 1)*n + 0][col] += 1   # c01 += p5
            W[(k_offset + c_co + 1)*n + 1][col] -= 1   # c11 -= p5
            col += 1
            
            # p6 = (a10-a00)(b00+b01)
            U[1*m + a_co][col] += 1
            U[0*m + a_co][col] -= 1
            V[b_ro*p + k_offset + b_co][col] += 1
            V[b_ro*p + k_offset + b_co + 1][col] += 1
            W[(k_offset + c_co)*n + 1][col] += 1       # c10 += p6
            col += 1
            
            # p7 = (a01-a11)(b10+b11)
            U[0*m + a_co + 1][col] += 1
            U[1*m + a_co + 1][col] -= 1
            V[(b_ro+1)*p + k_offset + b_co][col] += 1
            V[(b_ro+1)*p + k_offset + b_co + 1][col] += 1
            W[(k_offset + c_co + 1)*n + 1][col] += 1   # c11 += p7
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
