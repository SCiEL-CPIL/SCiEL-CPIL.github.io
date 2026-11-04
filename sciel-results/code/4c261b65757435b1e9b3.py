"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen-based decomposition for 4x4x4 blocks.
    # For 4x4x4: partition into 2x2 blocks, use Strassen (7 products) for the 
    # block-level multiplication, and schoolbook (4 scalar mults) for each 2x2x2.
    # Total rank for 4x4x4: 7 * 4 = 28
    # For 4x4x9: two 4x4x4 blocks (28 each) + one 4x4x1 schoolbook (16) = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    def add_strassen_2x2(U, V, W, a_row_offset, b_row_offset, c_row_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_row_offset: row offset in U for the 2x2 A block (4 rows)
        b_row_offset: row offset in V for the 2x2 B block (4 rows)
        c_row_offset: row offset in W for the 2x2 C block (4 rows)
        col_start: starting column index
        Returns: new column index
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 0][c] = 1
        W[c_row_offset + 2][c] = 1   # C21
        W[c_row_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_row_offset + 0][c] = 1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = -1
        W[c_row_offset + 1][c] = 1   # C12
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_row_offset + 3][c] = 1
        V[b_row_offset + 1][c] = 1
        V[b_row_offset + 0][c] = -1
        W[c_row_offset + 0][c] = 1   # C11
        W[c_row_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_row_offset + 0][c] = 1
        U[a_row_offset + 1][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = -1  # C11
        W[c_row_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_row_offset + 2][c] = 1
        U[a_row_offset + 0][c] = -1
        V[b_row_offset + 0][c] = 1
        V[b_row_offset + 1][c] = 1
        W[c_row_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_row_offset + 1][c] = 1
        U[a_row_offset + 3][c] = -1
        V[b_row_offset + 2][c] = 1
        V[b_row_offset + 3][c] = 1
        W[c_row_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    def add_4x4x4_strassen(U, V, W, a_offset, b_offset, c_offset, col_start):
        """
        Add 28-rank Strassen-based decomposition for 4x4x4.
        a_offset: row offset in U for the 4x4 A block (16 rows)
        b_offset: row offset in V for the 4x4 B block (16 rows)
        c_offset: row offset in W for the 4x4 C block (16 rows)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # For 4x4 matrix stored with row index = i*4 + j:
        # A11 = A[0:2, 0:2] -> indices 0,1,4,5
        # A12 = A[0:2, 2:4] -> indices 2,3,6,7
        # A21 = A[2:4, 0:2] -> indices 8,9,12,13
        # A22 = A[2:4, 2:4] -> indices 10,11,14,15
        
        # Strassen for 2x2 block matrix multiplication:
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
        
        # Each Mi is a 2x2 matrix product computed with schoolbook (rank 4)
        # We'll implement each Mi as 4 scalar multiplications
        
        # Helper to add schoolbook 2x2x2 for given linear combinations
        def add_schoolbook_2x2(U, V, W, u_indices, v_indices, w_indices, coeff_u, coeff_v, coeff_w, col):
            """
            Add 4 scalar multiplications for 2x2x2 schoolbook.
            u_indices: 4 indices into U (for the 2x2 A block)
            v_indices: 4 indices into V (for the 2x2 B block)
            w_indices: 4 indices into W (for the 2x2 C block)
            coeff_u: 4 coefficients for U
            coeff_v: 4 coefficients for V
            coeff_w: dict mapping (output_pos, input_pos) -> coeff for W
            col: starting column
            Returns: new column
            """
            for idx in range(4):
                U[u_indices[idx]][col] = coeff_u[idx]
                V[v_indices[idx]][col] = coeff_v[idx]
                # For schoolbook 2x2x2:
                # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
                # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
                # Each scalar mult contributes to one C entry
                # But here we're doing the whole 2x2 as 4 scalar mults
                # Actually for schoolbook 2x2x2, we have 4 scalar mults:
                # m0 = A00*B00, m1 = A01*B10, m2 = A00*B01, m3 = A01*B11
                # C00 = m0, C01 = m2, C10 = m1, C11 = m3
                # Wait no, that's not right. Let me reconsider.
                
                # Actually for 2x2x2 schoolbook:
                # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> needs 2 mults
                # This is getting confusing. Let me just directly implement.
                pass
            return col + 4
        
        # Let me directly implement the 28 scalar multiplications
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22 has entries:
        #   (0,0): A[0,0]+A[2,2] -> U[a_offset+0]=1, U[a_offset+10]=1
        #   (0,1): A[0,1]+A[2,3] -> U[a_offset+1]=1, U[a_offset+11]=1
        #   (1,0): A[1,0]+A[3,2] -> U[a_offset+4]=1, U[a_offset+14]=1
        #   (1,1): A[1,1]+A[3,3] -> U[a_offset+5]=1, U[a_offset+15]=1
        # B11+B22 has entries:
        #   (0,0): B[0,0]+B[2,2] -> V[b_offset+0]=1, V[b_offset+10]=1
        #   (0,1): B[0,1]+B[2,3] -> V[b_offset+1]=1, V[b_offset+11]=1
        #   (1,0): B[1,0]+B[3,2] -> V[b_offset+4]=1, V[b_offset+14]=1
        #   (1,1): B[1,1]+B[3,3] -> V[b_offset+5]=1, V[b_offset+15]=1
        
        # M1 is a 2x2 matrix, computed with schoolbook (4 scalar mults)
        # M1[0,0] = (A11+A22)[0,0]*(B11+B22)[0,0] + (A11+A22)[0,1]*(B11+B22)[1,0]
        # This is complex. Let me use a different approach.
        
        # I'll directly enumerate the 28 scalar multiplications.
        # For each of the 7 Strassen products, I need to compute a 2x2 matrix product.
        # A 2x2 matrix product C = A*B can be done with 4 scalar multiplications:
        # m0 = A[0,0]*B[0,0], m1 = A[0,1]*B[1,0], m2 = A[0,0]*B[0,1], m3 = A[0,1]*B[1,1]
        # C[0,0] = m0, C[0,1] = m2, C[1,0] = m1, C[1,1] = m3
        # Wait, that's only if we're doing outer product style. No, that's wrong.
        
        # Actually for 2x2 matrix multiplication C = A * B:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # This requires 8 scalar multiplications for schoolbook 2x2x2!
        # So 7 Strassen products * 8 = 56, not 28.
        
        # Hmm, I made an error. Let me reconsider.
        
        # For 2x2x2, the schoolbook rank is 8 (not 4).
        # So 7 * 8 = 56 for 4x4x4 with Strassen outer + schoolbook inner.
        
        # But we can do better by using Strassen for the inner 2x2x2 as well!
        # 7 * 7 = 49 for 4x4x4.
        
        # Let me use 49 for each 4x4x4 block.
        # For 4x4x9: 49 + 49 + 16 = 114
        
        # Actually, let me just use the simple schoolbook approach for now to ensure correctness.
        # Rank = 4*4*9 = 144
        
        return c
    
    # Given the complexity, let me just use schoolbook for correctness.
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
# EVOLVE-BLOCK-END
