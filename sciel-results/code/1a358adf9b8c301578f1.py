"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    # We use a Strassen-style decomposition on 2x2 blocks where possible.
    # The matrix multiplication is (3x4) * (4x8) -> (3x8).
    #
    # We partition the 3x4 matrix A into blocks and the 4x8 matrix B into blocks.
    # Since n=3, m=4, p=8.
    #
    # A is 3x4:
    #   Row 0: a00 a01 a02 a03
    #   Row 1: a10 a11 a12 a13
    #   Row 2: a20 a21 a22 a23
    #
    # B is 4x8:
    #   Columns 0-3 (left half) and columns 4-7 (right half)
    #   B_left is 4x4, B_right is 4x4
    #
    # C = A * B, C is 3x8.
    # C[:, 0:4] = A * B_left  (3x4 * 4x4 -> 3x4)
    # C[:, 4:8] = A * B_right (3x4 * 4x4 -> 3x4)
    #
    # For each 3x4 * 4x4 multiplication, we can split A into:
    #   A_top = A[0:2, :]  (2x4)
    #   A_bot = A[2, :]    (1x4)
    # And B into:
    #   B_top = B[0:2, :]  (2x4)
    #   B_bot = B[2:4, :]  (2x4)
    #
    # C_top = A_top * B  (2x4 * 4x4 -> 2x4)
    # C_bot = A_bot * B  (1x4 * 4x4 -> 1x4)
    #
    # For the 2x4 * 4x4 part, split B into two 2x2 blocks:
    #   B = [B11 B12; B21 B22] where each is 2x2
    #   A_top = [A1 A2] where A1, A2 are 2x2
    #
    # C_top = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    #
    # Each 2x2*2x2 uses Strassen: 7 multiplications instead of 4.
    #
    # Total multiplications:
    #   For one 3x4*4x4:
    #     C_top: 4 Strassen 2x2*2x2 = 4*7 = 28 mults
    #     C_bot: A_bot (1x4) * B (4x4)
    #       = A_bot[0]*B[:,0] + A_bot[1]*B[:,1] + A_bot[2]*B[:,2] + A_bot[3]*B[:,3]
    #       This is 4 dot products of length 4, each is 4 mults = 16 mults
    #     Total per 3x4*4x4: 28 + 16 = 44 mults
    #
    # Two such multiplications (left and right halves): 44*2 = 88 mults
    #
    # Schoolbook would be 3*4*8 = 96 mults.
    #
    # Let's build the tensor decomposition with rank 88.
    #
    # We need to express each scalar multiplication in the Strassen decomposition.
    #
    # Strassen for 2x2:
    #   Given X = [x11 x12; x21 x22], Y = [y11 y12; y21 y22]
    #   Z = X*Y = [z11 z12; z21 z22]
    #   M1 = (x11+x22)*(y11+y22)
    #   M2 = (x21+x22)*y11
    #   M3 = x11*(y12-y22)
    #   M4 = x22*(y21-y11)
    #   M5 = (x11+x12)*y22
    #   M6 = (x21-x11)*(y11+y12)
    #   M7 = (x12-x22)*(y21+y22)
    #   z11 = M1 + M4 - M5 + M7
    #   z12 = M3 + M5
    #   z21 = M2 + M4
    #   z22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a product of two linear combinations. In tensor decomposition,
    # each Mj contributes one rank-1 term. The U and V matrices encode the
    # coefficients of x and y in each linear combination, and W encodes
    # how each Mj contributes to the output.
    #
    # Let me index the 4x4 B matrix as B[i,j] where i in 0..3 (rows), j in 0..3 (cols).
    # B[i,j] corresponds to B[j*n + i] in the W convention? No.
    #
    # W[k*n+i, r]: k is the row of B (0..m-1=0..3), i is the row of A (0..n-1=0..2).
    # Wait, let me re-read the convention.
    #
    # The tensor T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    #
    # So W index is k*n + i where k is column of C (0..p-1=0..7) and i is row of C (0..n-1=0..2).
    # Wait, that doesn't seem right either. Let me think again.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the matrix multiplication tensor).
    #
    # U has shape (n*m, rank) = (12, rank). Index: i*m+j.
    # V has shape (m*p, rank) = (32, rank). Index: j*p+k.
    # W has shape (p*n, rank) = (24, rank). Index: k*n+i.
    #
    # So W[k*n+i, r] means the r-th rank-1 term contributes to C[i,k] with coefficient W[k*n+i, r].
    #
    # In a rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum over r to give T[i,j,k] = 1.
    #
    # For the schoolbook: each (i,j,k) triple gets its own column, with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    #
    # Now for Strassen, I need to identify which (i,j,k) triples are "multiplied together" in each Mj.
    #
    # Let me work through the 3x4*4x4 case (for the left half, columns 0-3 of B).
    #
    # A is 3x4, B_left is 4x4.
    # A_top = A[0:2, 0:4] (2x4), A_bot = A[2, 0:4] (1x4)
    # B_top = B[0:2, 0:4] (2x4), B_bot = B[2:4, 0:4] (2x4)
    #
    # C_top = A_top * B (2x4 * 4x4)
    #   = A_top[:, 0:2] * B_top + A_top[:, 2:4] * B_bot
    #
    # Let A1 = A_top[:, 0:2] (2x2), A2 = A_top[:, 2:4] (2x2)
    # B11 = B_top[:, 0:2] (2x2), B12 = B_top[:, 2:4] (2x2)
    # B21 = B_bot[:, 0:2] (2x2), B22 = B_bot[:, 2:4] (2x2)
    #
    # C_top[:, 0:2] = A1 * B11 + A2 * B21
    # C_top[:, 2:4] = A1 * B12 + A2 * B22
    #
    # Each 2x2*2x2 uses Strassen with 7 mults.
    #
    # So we have 4 Strassen blocks, each with 7 mults = 28 mults for C_top.
    #
    # C_bot = A_bot * B (1x4 * 4x4)
    #   C_bot[k] = sum_{j=0}^{3} A_bot[j] * B[j, k]
    #   This is 4 dot products, each with 4 terms = 16 mults.
    #
    # Total: 28 + 16 = 44 mults for 3x4*4x4.
    # Two halves: 88 mults.
    #
    # Now I need to enumerate each multiplication and create the corresponding U, V, W entries.
    #
    # Let me define the Strassen blocks.
    #
    # Block 1: A1 * B11 -> contributes to C[0:2, 0:2]
    #   A1[i,j] = A[i, j] for i in 0..1, j in 0..1
    #   B11[j,k] = B[j, k] for j in 0..1, k in 0..1
    #   C[i,k] += (A1*B11)[i,k] for i in 0..1, k in 0..1
    #
    # Block 2: A2 * B21 -> contributes to C[0:2, 0:2]
    #   A2[i,j] = A[i, 2+j] for i in 0..1, j in 0..1
    #   B21[j,k] = B[2+j, k] for j in 0..1, k in 0..1
    #   C[i,k] += (A2*B21)[i,k] for i in 0..1, k in 0..1
    #
    # Block 3: A1 * B12 -> contributes to C[0:2, 2:4]
    #   A1[i,j] = A[i, j] for i in 0..1, j in 0..1
    #   B12[j,k] = B[j, 2+k] for j in 0..1, k in 0..1
    #   C[i,k] += (A1*B12)[i,k] for i in 0..1, k in 2..3
    #
    # Block 4: A2 * B22 -> contributes to C[0:2, 2:4]
    #   A2[i,j] = A[i, 2+j] for i in 0..1, j in 0..1
    #   B22[j,k] = B[2+j, 2+k] for j in 0..1, k in 0..1
    #   C[i,k] += (A2*B22)[i,k] for i in 0..1, k in 2..3
    #
    # C_bot: A_bot[j] * B[j, k] for j in 0..3, k in 0..3
    #   C[2, k] += A_bot[j] * B[j, k]
    #   This is 16 individual scalar multiplications.
    #
    # Total: 4*7 + 16 = 44 mults.
    #
    # For the right half (columns 4-7 of B, i.e., B_right = B[:, 4:8]):
    # Same structure but B indices shifted by 4 in the column.
    #
    # Block 5: A1 * B11_right -> C[0:2, 4:6]
    #   B11_right[j,k] = B[j, 4+k]
    # Block 6: A2 * B21_right -> C[0:2, 4:6]
    #   B21_right[j,k] = B[2+j, 4+k]
    # Block 7: A1 * B12_right -> C[0:2, 6:8]
    #   B12_right[j,k] = B[j, 6+k]
    # Block 8: A2 * B22_right -> C[0:2, 6:8]
    #   B22_right[j,k] = B[2+j, 6+k]
    #
    # C_bot_right: A_bot[j] * B[j, 4+k] for j in 0..3, k in 0..3
    #   C[2, 4+k] += A_bot[j] * B[j, 4+k]
    #   16 mults.
    #
    # Total: 88 mults.
    #
    # Now let me code this up.
    
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, col_start):
        """
        Strassen 2x2*2x2 block.
        A block: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        B block: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        C block: rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1
        
        Returns the next column index.
        """
        nonlocal col
        # Strassen 7 multiplications:
        # M1 = (x11+x22)*(y11+y22)
        # M2 = (x21+x22)*y11
        # M3 = x11*(y12-y22)
        # M4 = x22*(y21-y11)
        # M5 = (x11+x12)*y22
        # M6 = (x21-x11)*(y11+y12)
        # M7 = (x12-x22)*(y21+y22)
        
        # Each Mj is a product of two linear combinations.
        # U encodes the A coefficients, V encodes the B coefficients, W encodes the C output.
        
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # U: A[0,0] coeff 1, A[1,1] coeff 1
        # V: B[0,0] coeff 1, B[1,1] coeff 1
        # W: C[0,0] += M1, C[1,1] += M1
        U[a_row_offset + 0][col] = 1
        U[a_row_offset + 1][col] = 1
        V[b_row_offset + 0][col] = 1
        V[b_row_offset + 1][col] = 1
        W[(c_col_offset + 0) * n + (c_row_offset + 0)][col] = 1
        W[(c_col_offset + 1) * n + (c_row_offset + 1)][col] = 1
        col += 1
        
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[a_row_offset + 1][col] = 1
        U[a_row_offset + 1][col] = 1  # A[1,0] and A[1,1] both have coeff 1? No.
        # Wait, A[1,0] is row 1, col 0 of the block. A[1,1] is row 1, col 1.
        # (x21+x22) means A[1,0] + A[1,1].
        U[a_row_offset + 1][col] = 1  # This is wrong, I'm overwriting.
        # Let me redo. I need to be more careful.
        
        # Let me restart this function with proper indexing.
        # A block indices: A[i,j] where i in 0..1, j in 0..1
        # Global A index: (a_row_offset + i) * m + (a_col_offset + j)
        # B block indices: B[i,j] where i in 0..1, j in 0..1
        # Global B index: (b_row_offset + i) * p + (b_col_offset + j)
        # C block indices: C[i,j] where i in 0..1, j in 0..1
        # Global C index: (c_col_offset + j) * n + (c_row_offset + i)
        
        # I already messed up col. Let me just not use this function and do it inline.
        return col
    
    # Let me do it differently. I'll just enumerate all 88 multiplications directly.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j, c):
        """
        Add Strassen decomposition for 2x2 block.
        A block starts at row a_i, col a_j (in global A indices)
        B block starts at row b_i, col b_j (in global B indices)
        C block starts at row c_i, col c_j (in global C indices)
        c is the current column index.
        Returns c + 7.
        """
        # A[i,j] global index: (a_i + i) * m + (a_j + j)
        # B[i,j] global index: (b_i + i) * p + (b_j + j)
        # C[i,j] global index: (c_j + j) * n + (c_i + i)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[(a_i+0)*m + (a_j+0)][c] = 1
        U[(a_i+1)*m + (a_j+1)][c] = 1
        V[(b_i+0)*p + (b_j+0)][c] = 1
        V[(b_i+1)*p + (b_j+1)][c] = 1
        W[(c_j+0)*n + (c_i+0)][c] = 1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[(a_i+1)*m + (a_j+0)][c] = 1
        U[(a_i+1)*m + (a_j+1)][c] = 1
        V[(b_i+0)*p + (b_j+0)][c] = 1
        W[(c_j+0)*n + (c_i+1)][c] = 1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[(a_i+0)*m + (a_j+0)][c] = 1
        V[(b_i+0)*p + (b_j+1)][c] = 1
        V[(b_i+1)*p + (b_j+1)][c] = -1
        W[(c_j+1)*n + (c_i+0)][c] = 1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[(a_i+1)*m + (a_j+1)][c] = 1
        V[(b_i+1)*p + (b_j+0)][c] = 1
        V[(b_i+0)*p + (b_j+0)][c] = -1
        W[(c_j+0)*n + (c_i+0)][c] = 1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[(a_i+0)*m + (a_j+0)][c] = 1
        U[(a_i+0)*m + (a_j+1)][c] = 1
        V[(b_i+1)*p + (b_j+1)][c] = 1
        W[(c_j+0)*n + (c_i+0)][c] = -1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[(a_i+1)*m + (a_j+0)][c] = 1
        U[(a_i+0)*m + (a_j+0)][c] = -1
        V[(b_i+0)*p + (b_j+0)][c] = 1
        V[(b_i+0)*p + (b_j+1)][c] = 1
        W[(c_j+1)*n + (c_i+1)][c] = 1
        c += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[(a_i+0)*m + (a_j+1)][c] = 1
        U[(a_i+1)*m + (a_j+1)][c] = -1
        V[(b_i+1)*p + (b_j+0)][c] = 1
        V[(b_i+1)*p + (b_j+1)][c] = 1
        W[(c_j+0)*n + (c_i+0)][c] = 1
        c += 1
        
        return c
    
    # Left half: B columns 0-3
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    
    # C_bot left: A[2, j] * B[j, k] for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    # Right half: B columns 4-7
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    col = add_strassen(0, 0, 0, 4, 0, 4, col)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    col = add_strassen(0, 2, 2, 4, 0, 4, col)
    # Block 7: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    col = add_strassen(0, 0, 0, 6, 0, 6, col)
    # Block 8: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    col = add_strassen(0, 2, 2, 6, 0, 6, col)
    
    # C_bot right: A[2, j] * B[j, k] for j in 0..3, k in 4..7
    for j in range(4):
        for k in range(4, 8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
