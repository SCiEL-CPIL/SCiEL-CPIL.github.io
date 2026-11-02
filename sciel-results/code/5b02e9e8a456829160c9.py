"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    # We use a decomposition that applies Strassen (rank 7) to a 2x2 block of the 4x4 x 4x4 part,
    # and schoolbook for the rest.
    # The 4x4 x 4x9 matrix multiplication can be seen as:
    # C (4x9) = A (4x4) * B (4x9)
    # We can view B as two 4x4 matrices plus one 4x1 matrix: B = [B1 | B2 | B3] where B1,B2 are 4x4, B3 is 4x1.
    # But Strassen works on 2x2 blocks.
    # Let's partition A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Let's partition B (4x9) into blocks. This is awkward.
    
    # Alternative: Use the known rank 128 decomposition for 4x4 x 4x4 and extend to 4x9?
    # Actually, the rank of the 4x4 x 4x4 tensor is at most 2 * 56 + 16 = 128 (using Strassen for 2x2 blocks recursively? No, Strassen is for 2x2. 4x4 = 2x (2x2) blocks?
    # 4x4 = 2x2 blocks of 2x2.
    # Multiplying two 4x4 matrices using 2x2 Strassen blocks:
    # We have 4 blocks of A and 4 blocks of B.
    # The multiplication is block matrix multiplication.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each Cij is 2x2. Each Aij, Bij is 2x2.
    # We can use Strassen for each 2x2 multiplication? No, the blocks are multiplied in a specific way.
    # Actually, we can use the Strassen algorithm on the 2x2 blocks.
    # The rank of the 4x4 x 4x4 tensor is at most 2^2 * 7 = 28? No.
    # If we treat the 4x4 matrices as 2x2 blocks of 2x2 matrices, the multiplication is:
    # C = A * B where A,B are 4x4.
    # We can write A and B as 2x2 matrices whose entries are 2x2 matrices.
    # Then C is a 2x2 matrix of 2x2 matrices.
    # C11 = A11 B11 + A12 B21
    # This is not a single Strassen step.
    
    # Let's stick to the rank 128 decomposition mentioned in the comment:
    # "2x (4x4x4 Strassen-block rank 56) + 1x (4x4x1 Schoolbook rank 16)"
    # Wait, 4x4x4 Strassen-block rank 56?
    # The rank of the 2x2x2 tensor is 7.
    # The rank of the 4x4x4 tensor is at most 4^2 * 7 = 112? Or 2^3 * 7 = 56?
    # If we use Strassen recursively: 4x4 = 2x2 blocks of 2x2.
    # The multiplication of two 4x4 matrices using Strassen on the 2x2 block structure:
    # We have 4 2x2 blocks.
    # The number of scalar multiplications is 2^3 * 7 = 56.
    # So rank 56 for 4x4 x 4x4.
    # For 4x4 x 4x9:
    # We can view the 4x9 matrix as three 4x3 matrices? No.
    # Let's use the schoolbook for the full 4x4 x 4x9: rank = 4*4*9 = 144.
    # The rank 128 decomposition:
    # Split the 4x9 matrix into two 4x4 matrices and one 4x1 matrix.
    # B = [B1 | B2 | B3] where B1, B2 are 4x4, B3 is 4x1.
    # C = A * B = [A*B1 | A*B2 | A*B3]
    # A*B1 and A*B2 are 4x4 multiplications. A*B3 is 4x4 x 4x1 multiplication.
    # Rank of A*B1 (4x4x4) <= 56.
    # Rank of A*B2 (4x4x4) <= 56.
    # Rank of A*B3 (4x4x1) = 4*4*1 = 16 (schoolbook).
    # Total rank = 56 + 56 + 16 = 128.
    
    rank = 128
    U = [[0] * rank for _ in range(n * m)]  # 16 x 128
    V = [[0] * rank for _ in range(m * p)]  # 36 x 128
    W = [[0] * rank for _ in range(p * n)]  # 36 x 128
    
    col_offset = 0
    
    # Helper to add a 2x2 Strassen decomposition
    def add_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are the global indices for the 2x2 blocks
        # A block: A[a_rows[0]:a_rows[1], a_cols[0]:a_cols[1]] etc.
        # Actually, let's define the indices for the 4 elements of a 2x2 block.
        # A2x2 indices: (r0,c0), (r0,c1), (r1,c0), (r1,c1)
        # In global 4x4 matrix, row i, col j is index i*4+j
        # Let's define a function to get the global index for a 2x2 block element
        def get_idx(r, c, base_r, base_c):
            return (base_r + r) * 4 + (base_c + c)
        
        # A elements
        a00 = a_rows[0] * 4 + a_cols[0]
        a01 = a_rows[0] * 4 + a_cols[1]
        a10 = a_rows[1] * 4 + a_cols[0]
        a11 = a_rows[1] * 4 + a_cols[1]
        
        # B elements
        b00 = b_rows[0] * 4 + b_cols[0]
        b01 = b_rows[0] * 4 + b_cols[1]
        b10 = b_rows[1] * 4 + b_cols[0]
        b11 = b_rows[1] * 4 + b_cols[1]
        
        # C elements
        c00 = c_rows[0] * 4 + c_cols[0]
        c01 = c_rows[0] * 4 + c_cols[1]
        c10 = c_rows[1] * 4 + c_cols[0]
        c11 = c_rows[1] * 4 + c_cols[1]
        
        # Strassen M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
        U[a00][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        V[b11][offset] += 1
        W[c00][offset] += 1
        W[c11][offset] += 1
        offset += 1
        
        # M2 = (a10+a11)b00 -> C10+=1, C11-=-1
        U[a10][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        W[c10][offset] += 1
        W[c11][offset] -= 1
        offset += 1
        
        # M3 = a00(b01-b11) -> C01+=1, C11+=1
        U[a00][offset] += 1
        V[b01][offset] += 1
        V[b11][offset] -= 1
        W[c01][offset] += 1
        W[c11][offset] += 1
        offset += 1
        
        # M4 = a11(b10-a10) -> C00+=1, C10+=1
        # Wait, standard Strassen M4 = d(g-c). d=a11, g=b10, c=a10.
        # M4 = a11 * (b10 - a10) is not bilinear.
        # Correct Strassen M4 = a11 * b10. And we adjust other terms.
        # Let's use the correct formulas:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-a10) -> This is wrong.
        # Correct: M4 = a11 * b10.
        # Then C00 = M1 - M4 - M2 + M5?
        # Let's just use the standard 7 products and the reconstruction formulas.
        # C00 = M1 - M4 - M2 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M4 + M6
        # Where:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11 * b10
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # Let's restart the Strassen addition with correct formulas.
        # We will undo the previous 4 terms and do it correctly.
        # For simplicity, let's just use schoolbook for the 2x2 blocks to ensure correctness, but that would be rank 16 for 2x2, so 16*16 = 256 total? No.
        # Schoolbook for 2x2 is rank 4. 16 blocks of 2x2? No, 4x4 = 4 2x2 blocks.
        # If we use schoolbook for each 2x2 multiplication, we have 16 multiplications of 2x2? No.
        # The block multiplication C11 = A11 B11 + A12 B21.
        # This is 2 multiplications of 2x2 matrices.
        # If we use schoolbook for 2x2, each is rank 4.
        # Total multiplications: 8 (4 blocks, 2 mults each).
        # Rank = 8 * 4 = 32 for the 4x4 x 4x4 part? No, that's not how it works.
        
        # Let's just use the rank 128 decomposition with the correct Strassen formulas.
        # I will implement the 7 Strassen products correctly.
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        V[b11][offset] += 1
        # C00 += M1, C11 += M1
        W[c00][offset] += 1
        W[c11][offset] += 1
        offset += 1
        
        # M2 = (a10+a11)b00
        U[a10][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        # C10 += M2, C11 -= M2
        W[c10][offset] += 1
        W[c11][offset] -= 1
        offset += 1
        
        # M3 = a00(b01-b11)
        U[a00][offset] += 1
        V[b01][offset] += 1
        V[b11][offset] -= 1
        # C01 += M3, C11 += M3
        W[c01][offset] += 1
        W[c11][offset] += 1
        offset += 1
        
        # M4 = a11*b10
        U[a11][offset] += 1
        V[b10][offset] += 1
        # C00 -= M4, C10 += M4
        W[c00][offset] -= 1
        W[c10][offset] += 1
        offset += 1
        
        # M5 = (a00+a01)b11
        U[a00][offset] += 1
        U[a01][offset] += 1
        V[b11][offset] += 1
        # C00 += M5, C01 += M5
        W[c00][offset] += 1
        W[c01][offset] += 1
        offset += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][offset] += 1
        U[a00][offset] -= 1
        V[b00][offset] += 1
        V[b01][offset] += 1
        # C11 += M6
        W[c11][offset] += 1
        offset += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][offset] += 1
        U[a11][offset] -= 1
        V[b10][offset] += 1
        V[b11][offset] += 1
        # C00 += M7? No.
        # C00 = M1 - M4 - M2 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M4 + M6
        # Wait, M7 is not used in C?
        # Standard Strassen:
        # C00 = M1 - M4 - M2 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M4 + M6
        # M7 is not used? That can't be right.
        # Let's check the formulas again.
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 - M2 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M4 + M6
        # M7 is used in C00? No.
        # I will not implement Strassen due to complexity and risk of error.
        # I will use schoolbook for the entire 4x4 x 4x9, which is rank 144.
        # This ensures validity.
        
        pass
    
    # Use schoolbook for the entire matrix multiplication
    rank = n * m * p  # 4*4*9 = 144
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
# EVOLVE-BLOCK-END
