"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2 * 2x2 matrix multiplication, rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Strassen's 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]  # A indices: i*2+j
        V = [[0]*7 for _ in range(4)]  # B indices: j*2+k
        W = [[0]*7 for _ in range(4)]  # C indices: k*2+i
        
        # M1: (A11+A22)(B11+B22) -> C11, C22
        # A11: i=0,j=0; A22: i=1,j=1
        # B11: j=0,k=0; B22: j=1,k=1
        U[0][0] = 1; U[3][0] = 1  # A11, A22
        V[0][0] = 1; V[3][0] = 1  # B11, B22
        W[0][0] = 1; W[3][0] = 1  # C11(k=0,i=0), C22(k=1,i=1)
        
        # M2: (A21+A22)B11 -> C21, C22
        # A21: i=1,j=0; A22: i=1,j=1
        # B11: j=0,k=0
        U[2][1] = 1; U[3][1] = 1  # A21, A22
        V[0][1] = 1  # B11
        W[2][1] = 1; W[3][1] = -1  # C21(k=0,i=1), C22(k=1,i=1)
        
        # M3: A11(B21-B22) -> C12, C22
        # A11: i=0,j=0
        # B21: j=1,k=0; B22: j=1,k=1
        U[0][2] = 1  # A11
        V[2][2] = 1; V[3][2] = -1  # B21, B22
        W[1][2] = 1; W[3][2] = 1  # C12(k=1,i=0), C22(k=1,i=1)
        
        # M4: A22(B12-B11) -> C11, C21
        # A22: i=1,j=1
        # B12: j=0,k=1; B11: j=0,k=0
        U[3][3] = 1  # A22
        V[1][3] = 1; V[0][3] = -1  # B12, B11
        W[0][3] = 1; W[2][3] = 1  # C11(k=0,i=0), C21(k=0,i=1)
        
        # M5: (A11+A12)B22 -> C11, C12
        # A11: i=0,j=0; A12: i=0,j=1
        # B22: j=1,k=1
        U[0][4] = 1; U[1][4] = 1  # A11, A12
        V[3][4] = 1  # B22
        W[0][4] = -1; W[1][4] = 1  # C11(k=0,i=0), C12(k=1,i=0)
        
        # M6: (A21-A11)(B11+B12) -> C22
        # A21: i=1,j=0; A11: i=0,j=0
        # B11: j=0,k=0; B12: j=0,k=1
        U[2][5] = 1; U[0][5] = -1  # A21, A11
        V[0][5] = 1; V[1][5] = 1  # B11, B12
        W[3][5] = 1  # C22(k=1,i=1)
        
        # M7: (A12-A22)(B21+B22) -> C11
        # A12: i=0,j=1; A22: i=1,j=1
        # B21: j=1,k=0; B22: j=1,k=1
        U[1][6] = 1; U[3][6] = -1  # A12, A22
        V[2][6] = 1; V[3][6] = 1  # B21, B22
        W[0][6] = 1  # C11(k=0,i=0)
        
        return U, V, W
    
    def strassen_4x4x4():
        """Use 2x2x2 Strassen recursively to get 4x4 * 4x4 rank 49.
        Returns (U, V, W) with U: (16, 49), V: (16, 49), W: (16, 49)
        """
        U2, V2, W2 = strassen_2x2x2()
        
        # For 4x4 * 4x4, split into 2x2 blocks
        # A is 4x4, B is 4x4, C is 4x4
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = A*B
        
        # We apply Strassen at the block level, then within each block multiplication
        # use the 2x2x2 Strassen decomposition.
        
        # Block-level Strassen:
        # M1_block = (A11+A22)(B11+B22)
        # M2_block = (A21+A22)B11
        # M3_block = A11(B21-B22)
        # M4_block = A22(B12-B11)
        # M5_block = (A11+A12)B22
        # M6_block = (A21-A11)(B11+B12)
        # M7_block = (A12-A22)(B21+B22)
        
        # Each block multiplication is 2x2 * 2x2, which has rank 7.
        # So total rank = 7 * 7 = 49.
        
        # For each block-level multiplication, we need to:
        # 1. Form the appropriate linear combinations of A and B blocks
        # 2. Multiply using 2x2x2 Strassen (rank 7)
        
        # A block indices: A11 rows 0-1, cols 0-1; A12 rows 0-1, cols 2-3
        #                 A21 rows 2-3, cols 0-1; A22 rows 2-3, cols 2-3
        # B block indices: B11 rows 0-1, cols 0-1; B12 rows 0-1, cols 2-3
        #                 B21 rows 2-3, cols 0-1; B22 rows 2-3, cols 2-3
        
        # U index for 4x4: i*4+j where i in [0,3], j in [0,3]
        # V index for 4x4: j*4+k where j in [0,3], k in [0,3]
        # W index for 4x4: k*4+i where k in [0,3], i in [0,3]
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # For each of the 7 block-level multiplications, we use 7 sub-multiplications
        # Total: 49 columns
        
        # Define the 7 block-level combinations
        # Each entry is (coeff_A_block, coeff_B_block) for the product
        # M1: (A11+A22) * (B11+B22)
        # M2: (A21+A22) * B11
        # M3: A11 * (B21-B22)
        # M4: A22 * (B12-B11)
        # M5: (A11+A12) * B22
        # M6: (A21-A11) * (B11+B12)
        # M7: (A12-A22) * (B21+B22)
        
        # Block A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # Block B indices: 0=B11, 1=B12, 2=B21, 3=B22
        
        block_defs = [
            # (A_block_coeffs, B_block_coeffs, C_block_output)
            # C_block: 0=C11, 1=C12, 2=C21, 3=C22
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1 -> C11, C22
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2 -> C21, C22
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # M3 -> C12, C22
            ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),   # M4 -> C11, C21
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5 -> C11, C12
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6 -> C22
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7 -> C11
        ]
        
        # For each block-level multiplication, apply 2x2x2 Strassen
        # A2, V2, W2 are the 2x2x2 Strassen factors
        # A2[i2*2+j2], V2[j2*2+k2], W2[k2*2+i2]
        
        # Block A: block bi, local indices (i2, j2) -> global (i2 + 2*bi_row, j2 + 2*bi_col)
        # Block B: block bj, local indices (j2, k2) -> global (j2 + 2*bj_row, k2 + 2*bj_col)
        # Block C: block bc, local indices (k2, i2) -> global (k2 + 2*bc_col, i2 + 2*bc_row)
        
        # Block row/col: 0,0 -> top-left; 0,1 -> top-right; 1,0 -> bottom-left; 1,1 -> bottom-right
        block_positions = {
            0: (0, 0),  # top-left
            1: (0, 1),  # top-right
            2: (1, 0),  # bottom-left
            3: (1, 1),  # bottom-right
        }
        
        for bi in range(7):
            A_coeffs, B_coeffs, C_coeffs = block_defs[bi]
            
            for sub in range(7):
                col = bi * 7 + sub
                
                # For each A block with non-zero coeff
                for a_block in range(4):
                    if A_coeffs[a_block] == 0:
                        continue
                    a_row, a_col = block_positions[a_block]
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i2 + 2 * a_row
                            j = j2 + 2 * a_col
                            U[i*4 + j][col] += A_coeffs[a_block] * U2[i2*2 + j2][sub]
                
                # For each B block with non-zero coeff
                for b_block in range(4):
                    if B_coeffs[b_block] == 0:
                        continue
                    b_row, b_col = block_positions[b_block]
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j2 + 2 * b_row
                            k = k2 + 2 * b_col
                            V[j*4 + k][col] += B_coeffs[b_block] * V2[j2*2 + k2][sub]
                
                # For each C block with non-zero coeff
                for c_block in range(4):
                    if C_coeffs[c_block] == 0:
                        continue
                    c_row, c_col = block_positions[c_block]
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k2 + 2 * c_col
                            i = i2 + 2 * c_row
                            W[k*4 + i][col] += C_coeffs[c_block] * W2[k2*2 + i2][sub]
        
        return U, V, W
    
    def schoolbook_4x4x5():
        """Schoolbook decomposition for 4x4 * 4x5, rank 80."""
        n, m, p = 4, 4, 5
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
        return U, V, W
    
    # Strategy: Use Strassen for the first 4 columns (4x4 * 4x4, rank 49)
    # and schoolbook for the 5th column (4x4 * 4x1, rank 16)
    # Total rank: 49 + 16 = 65
    
    if time.time() > deadline - 5:
        U, V, W = schoolbook_4x4x5()
        return {"U": U, "V": V, "W": W}
    
    try:
        U_str, V_str, W_str = strassen_4x4x4()
        
        # Now handle the 5th column
        # For the 5th column (k=4), we need:
        # sum_c U[i*4+j, c] * V[j*5+4, c] * W[4*4+i, c] = 1 for all i,j
        
        # We can add 16 more terms for the 5th column using schoolbook
        rank_total = 49 + 16
        U = [[0] * rank_total for _ in range(16)]
        V = [[0] * rank_total for _ in range(20)]  # m*p = 4*5 = 20
        W = [[0] * rank_total for _ in range(20)]  # p*n = 5*4 = 20
        
        # Copy Strassen decomposition for first 4 columns
        for row in range(16):
            for col in range(49):
                U[row][col] = U_str[row][col]
        
        for row in range(16):  # j*4+k where k < 4
            for col in range(49):
                V[row][col] = V_str[row][col]
        
        for row in range(16):  # k*4+i where k < 4
            for col in range(49):
                W[row][col] = W_str[row][col]
        
        # Add schoolbook terms for 5th column (k=4)
        col_offset = 49
        for i in range(4):
            for j in range(4):
                U[i*4 + j][col_offset] = 1
                V[j*5 + 4][col_offset] = 1
                W[4*4 + i][col_offset] = 1
                col_offset += 1
        
        # Verify the decomposition
        # Check: sum_c U[i*4+j, c] * V[j*5+k, c] * W[k*4+i, c] = 1 for all i,j,k
        error = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    val = 0
                    for c in range(rank_total):
                        val += U[i*4+j][c] * V[j*5+k][c] * W[k*4+i][c]
                    if val != 1:
                        error += abs(val - 1)
        
        if error == 0:
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W = schoolbook_4x4x5()
            return {"U": U, "V": V, "W": W}
    
    except Exception:
        U, V, W = schoolbook_4x4x5()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
