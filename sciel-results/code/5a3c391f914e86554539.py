import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Use the rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    The 4x4x8 problem is equivalent to two independent 4x4x4 problems (for the first 4
    columns and last 4 columns of B). So total rank = 49 + 49 = 98.
    
    The rank-49 decomposition for 4x4x4 is obtained by applying Strassen's algorithm
    to 2x2 blocks of 4x4 matrices. Each 4x4 matrix is viewed as a 2x2 block matrix
    of 2x2 matrices. Strassen reduces the 8 scalar multiplications needed for 2x2
    matrix multiplication to 7, so for 4x4 we get 7 * 7 = 49 scalar multiplications.
    """
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Build the rank-49 decomposition for 4x4x4 matrix multiplication
    # using Strassen's algorithm applied to 2x2 blocks.
    
    # First, define the Strassen factors for 2x2x2 (rank 7)
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: (i,j) with i,j in {0,1} -> flat index i*2+j
    # B indices: (j,k) with j,k in {0,1} -> flat index j*2+k
    # C indices: (k,i) with k,i in {0,1} -> flat index k*2+i
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x2, the factors are:
    # U (4x7): rows are A entries (00, 01, 10, 11)
    # V (4x7): rows are B entries (00, 01, 10, 11)
    # W (4x7): rows are C entries (00, 01, 10, 11) in trace convention C[k*n+i] = C[i][k]
    
    # Wait, let me be careful about the index conventions.
    # For the full 4x4x4 problem:
    # A is 4x4: index (i,j) -> flat i*m+j = i*4+j
    # B is 4x4: index (j,k) -> flat j*p+k = j*4+k
    # C is 4x4: index (k,i) -> flat k*n+i = k*4+i (trace convention)
    
    # For 2x2x2 sub-problem:
    # A is 2x2: index (i,j) -> flat i*2+j
    # B is 2x2: index (j,k) -> flat j*2+k
    # C is 2x2: index (k,i) -> flat k*2+i
    
    # Let me define the Strassen factors for 2x2x2.
    
    # U_222 (4, 7):
    # Row 0 (A00): appears in M1, M3, M5, M6
    # Row 1 (A01): appears in M3, M5, M7
    # Row 2 (A10): appears in M2, M6
    # Row 3 (A11): appears in M1, M2, M4, M7
    
    U_222 = [
        [1, 0, 1, 0, 1, 1, 0],  # A00
        [0, 0, 1, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, -1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ]
    
    # V_222 (4, 7):
    # Row 0 (B00): appears in M1, M2, M4, M6
    # Row 1 (B01): appears in M3, M6
    # Row 2 (B10): appears in M4, M6, M7
    # Row 3 (B11): appears in M1, M3, M5, M7
    
    V_222 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, -1, 0, 0, 1, 0], # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, 1, 0, 1, 0, 1],  # B11
    ]
    
    # W_222 (4, 7):
    # Row 0 (C00): M1 + M4 - M5 + M7
    # Row 1 (C01): M3 + M5
    # Row 2 (C10): M2 + M4
    # Row 3 (C11): M1 - M2 + M3 + M6
    
    W_222 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]
    
    # Now, for 4x4x4, we view each 4x4 matrix as a 2x2 block matrix of 2x2 matrices.
    # A_4x4 = [[A00_2x2, A01_2x2],
    #          [A10_2x2, A11_2x2]]
    # B_4x4 = [[B00_2x2, B01_2x2],
    #          [B10_2x2, B11_2x2]]
    # C_4x4 = [[C00_2x2, C01_2x2],
    #          [C10_2x2, C11_2x2]]
    
    # Using Strassen at the block level:
    # M1 = (A00+A11)(B00+B11) -> 7 scalar mults for each 2x2 product
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B01)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 matrix product, which requires 7 scalar multiplications.
    # So total scalar multiplications = 7 * 7 = 49.
    
    # Now I need to build the 4x4x4 factors (U, V, W) of shape (16, 49).
    
    # Let me index the 49 scalar multiplications by (block_mult, scalar_mult)
    # where block_mult in {0..6} and scalar_mult in {0..6}.
    
    # For each block multiplication Mj (j=0..6), we need to:
    # 1. Form the appropriate linear combinations of A blocks for the first factor
    # 2. Form the appropriate linear combinations of B blocks for the second factor
    # 3. Apply the 2x2x2 Strassen decomposition to get the 7 scalar multiplications
    # 4. Combine the results to form the C blocks
    
    # Let me define the block-level Strassen coefficients.
    # For block Mj, we have:
    # First factor: sum over (i,j) of coeff_A[j][i][block_i][block_j] * A_block_i_j
    # Second factor: sum over (k,l) of coeff_B[j][k][block_k][block_l] * B_block_k_l
    
    # Actually, let me think about this more carefully.
    
    # At the block level, we have 2x2 block matrices. Strassen gives us 7 block
    # multiplications. For each block multiplication, we need to form linear
    # combinations of the 2x2 sub-blocks.
    
    # Let me define for each of the 7 block multiplications:
    # - The coefficients for combining A blocks (2x2 grid of 2x2 matrices)
    # - The coefficients for combining B blocks (2x2 grid of 2x2 matrices)
    # - The coefficients for combining the resulting C blocks
    
    # Block multiplication 0 (M1): (A00+A11)(B00+B11)
    # A combination: A00 + A11
    # B combination: B00 + B11
    # C contribution: C00 += M1, C11 += M1
    
    # Block multiplication 1 (M2): (A10+A11)B00
    # A combination: A10 + A11
    # B combination: B00
    # C contribution: C10 += M2, C11 -= M2
    
    # Block multiplication 2 (M3): A00(B11-B01)
    # A combination: A00
    # B combination: B11 - B01
    # C contribution: C01 += M3, C11 += M3
    
    # Block multiplication 3 (M4): A11(B10-B00)
    # A combination: A11
    # B combination: B10 - B00
    # C contribution: C00 += M4, C10 += M4
    
    # Block multiplication 4 (M5): (A00+A01)B11
    # A combination: A00 + A01
    # B combination: B11
    # C contribution: C00 -= M5, C01 += M5
    
    # Block multiplication 5 (M6): (A10-A00)(B00+B01)
    # A combination: A10 - A00
    # B combination: B00 + B01
    # C contribution: C11 += M6
    
    # Block multiplication 6 (M7): (A01-A11)(B10+B11)
    # A combination: A01 - A11
    # B combination: B10 + B11
    # C contribution: C00 += M7
    
    # Now, for each block multiplication j, we apply the 2x2x2 Strassen decomposition.
    # The 2x2x2 Strassen gives us 7 scalar multiplications.
    
    # For block multiplication j, the first factor is a linear combination of A blocks.
    # Each A block is a 2x2 matrix. So the first factor for block mult j is a 2x2 matrix
    # formed by combining the 2x2 sub-blocks.
    
    # Let me define the A coefficients for each block multiplication.
    # A_coeffs[j][bi][bj] = coefficient for A block (bi, bj) in block mult j
    # where bi, bj in {0,1}
    
    A_coeffs = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, -1], [0, 1]],  # M3: B11 - B01
        [[-1, 0], [1, 0]],  # M4: B10 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 0], [1, 1]],   # M7: B10 + B11
    ]
    
    # C coefficients: for each block mult j, which C blocks get which coefficient
    # C_coeffs[j][ci][cj] = coefficient for C block (ci, cj) from block mult j
    # where ci, cj in {0,1}
    
    C_coeffs = [
        [[1, 0], [0, 1]],   # M1: C00 += M1, C11 += M1
        [[0, 0], [1, -1]],  # M2: C10 += M2, C11 -= M2
        [[0, 1], [0, 1]],   # M3: C01 += M3, C11 += M3
        [[1, 0], [1, 0]],   # M4: C00 += M4, C10 += M4
        [[-1, 1], [0, 0]],  # M5: C00 -= M5, C01 += M5
        [[0, 0], [0, 1]],   # M6: C11 += M6
        [[1, 0], [0, 0]],   # M7: C00 += M7
    ]
    
    # Now build the 4x4x4 factors.
    # U_444 (16, 49), V_444 (16, 49), W_444 (16, 49)
    
    U_444 = [[0] * 49 for _ in range(16)]
    V_444 = [[0] * 49 for _ in range(16)]
    W_444 = [[0] * 49 for _ in range(16)]
    
    # For each block multiplication j (0..6) and each scalar multiplication s (0..6):
    # The scalar multiplication index is j*7 + s
    
    for j in range(7):
        for s in range(7):
            col = j * 7 + s
            
            # For U: combine A blocks according to A_coeffs[j], then apply U_222[s]
            # A block (bi, bj) has flat index (bi*2 + i) * 4 + (bj*2 + j2) for i,j2 in {0,1}
            # Wait, let me be more careful.
            
            # A is 4x4. A[i][j] where i,j in {0,1,2,3}.
            # Flat index = i*4 + j.
            # Block (bi, bj) contains A[bi*2+i2][bj*2+j2] for i2,j2 in {0,1}.
            # So A[bi*2+i2][bj*2+j2] has flat index (bi*2+i2)*4 + (bj*2+j2).
            
            # For block mult j, the first factor is:
            # sum_{bi,bj} A_coeffs[j][bi][bj] * A_block(bi,bj)
            # Each A_block(bi,bj) is a 2x2 matrix.
            # The (i2,j2) entry of this combined 2x2 matrix is:
            # sum_{bi,bj} A_coeffs[j][bi][bj] * A[bi*2+i2][bj*2+j2]
            
            # Now apply the 2x2x2 Strassen: the s-th scalar multiplication uses
            # U_222[i2*2+j2][s] for the A part.
            
            # So for U_444[flat_A_index][col]:
            # flat_A_index = (bi*2+i2)*4 + (bj*2+j2)
            # U_444[flat_A_index][col] += A_coeffs[j][bi][bj] * U_222[i2*2+j2][s]
            
            for bi in range(2):
                for bj in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            flat_A = (bi * 2 + i2) * 4 + (bj * 2 + j2)
                            U_444[flat_A][col] += A_coeffs[j][bi][bj] * U_222[i2 * 2 + j2][s]
            
            # For V: similar with B
            # B is 4x4. B[j][k] where j,k in {0,1,2,3}.
            # Flat index = j*4 + k.
            # Block (bk, bl) contains B[bk*2+j2][bl*2+k2] for j2,k2 in {0,1}.
            # Wait, I need to be careful. B is indexed by (j,k) where j is the row
            # (corresponding to the column of A) and k is the column.
            
            # B block (bk, bl) means B rows bk*2+j2 and columns bl*2+k2.
            # Flat index = (bk*2+j2)*4 + (bl*2+k2).
            
            for bk in range(2):
                for bl in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            flat_B = (bk * 2 + j2) * 4 + (bl * 2 + k2)
                            V_444[flat_B][col] += B_coeffs[j][bk][bl] * V_222[j2 * 2 + k2][s]
            
            # For W: combine C blocks according to C_coeffs[j], then apply W_222[s]
            # C is 4x4. C[k][i] in trace convention, flat index = k*4 + i.
            # Block (ck, ci) contains C[ck*2+k2][ci*2+i2] for k2,i2 in {0,1}.
            # Flat index = (ck*2+k2)*4 + (ci*2+i2).
            
            for ck in range(2):
                for ci in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            flat_C = (ck * 2 + k2) * 4 + (ci * 2 + i2)
                            W_444[flat_C][col] += C_coeffs[j][ck][ci] * W_222[k2 * 2 + i2][s]
    
    # Now build the 4x4x8 factors.
    # The 4x4x8 problem: A is 4x4, B is 4x8, C is 4x8.
    # C[k][i] in trace convention, flat index = k*8 + i.
    # B is 4x8: B[j][k] where j in {0..3}, k in {0..7}. Flat index = j*8 + k.
    
    # We split B into B_left (columns 0-3) and B_right (columns 4-7).
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4).
    # C = [C_left, C_right].
    
    # For the left part, we use the 4x4x4 decomposition.
    # For the right part, we use another copy.
    
    # Total rank = 49 + 49 = 98.
    
    U_448 = [[0] * 98 for _ in range(16)]
    V_448 = [[0] * 98 for _ in range(32)]
    W_448 = [[0] * 98 for _ in range(32)]
    
    # Left part (columns 0-3 of B, columns 0-3 of C):
    for col in range(49):
        # U
        for row in range(16):
            U_448[row][col] = U_444[row][col]
        
        # V: B_left[j][k] = B[j][k] for k in {0..3}
        # Flat index in B_448 = j*8 + k
        # Flat index in B_444 = j*4 + k
        for j in range(4):
            for k in range(4):
                flat_B_448 = j * 8 + k
                flat_B_444 = j * 4 + k
                V_448[flat_B_448][col] = V_444[flat_B_444][col]
        
        # W: C_left[k][i] = C[k][i] for k in {0..3}
        # Flat index in C_448 = k*8 + i
        # Flat index in C_444 = k*4 + i
        for k in range(4):
            for i in range(4):
                flat_C_448 = k * 8 + i
                flat_C_444 = k * 4 + i
                W_448[flat_C_448][col] = W_444[flat_C_444][col]
    
    # Right part (columns 4-7 of B, columns 4-7 of C):
    for col in range(49):
        col_offset = 49 + col
        
        # U
        for row in range(16):
            U_448[row][col_offset] = U_444[row][col]
        
        # V: B_right[j][k] = B[j][k+4] for k in {0..3}
        # Flat index in B_448 = j*8 + (k+4)
        # Flat index in B_444 = j*4 + k
        for j in range(4):
            for k in range(4):
                flat_B_448 = j * 8 + (k + 4)
                flat_B_444 = j * 4 + k
                V_448[flat_B_448][col_offset] = V_444[flat_B_444][col]
        
        # W: C_right[k][i] = C[k][i+4] for k in {0..3}
        # Flat index in C_448 = k*8 + (i+4)
        # Flat index in C_444 = k*4 + i
        for k in range(4):
            for i in range(4):
                flat_C_448 = k * 8 + (i + 4)
                flat_C_444 = k * 4 + i
                W_448[flat_C_448][col_offset] = W_444[flat_C_444][col]
    
    return {"U": U_448, "V": V_448, "W": W_448}
