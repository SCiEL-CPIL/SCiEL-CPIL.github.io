"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = 150  # seconds

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Strassen-based decomposition for 4x4x4, then extend to 4x4x6
    # Strassen for 2x2x2 has rank 7 instead of 8
    # For 4x4x4, we can use 2x2x2 Strassen blocks: rank = 4*7 = 28 (vs schoolbook 64)
    # For 4x4x6, we need to handle the 6 columns of the second matrix
    # Split 4x4x6 into two 4x4x3 blocks
    # For 4x4x3: we can think of it as 4x4x4 with 3 columns zeroed, but that doesn't directly help
    # Alternative: split the 6-wide matrix into 4+2, use 4x4x4 Strassen (rank 28) and 4x4x2 
    # 4x4x2: schoolbook rank = 32, or we can be smarter
    # Actually, let's use a direct approach:
    # Split 6 = 2 + 2 + 2, use three 4x4x2 multiplications
    # 4x4x2: we can view this as computing C = A*B where A is 4x4, B is 4x2
    # Each column of C is a 4x1 vector, each being A times a 4x1 column
    # For 4x1x2 (i.e., 4x4 times 4x2), schoolbook is 4*4*2 = 32
    # Can we do better? 4x4x2 = 32 is likely tight for this small case
    # Total: 3 * 32 = 96... that's worse than schoolbook 96. Same.
    
    # Let's try: 6 = 4 + 2
    # 4x4x4 part: use Strassen 2x2x2 blocks -> rank 28
    # 4x4x2 part: schoolbook -> rank 32
    # Total: 28 + 32 = 60
    
    # Strassen 2x2x2 decomposition:
    # Given A (2x2), B (2x2), C = A*B
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4 using 2x2x2 Strassen:
    # Split A into four 2x2 blocks: A00, A01, A10, A11 (row-major, 0=top-left)
    # Similarly B into B00, B01, B10, B11
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each 2x2x2 multiplication uses 7 terms, 4 such multiplications = 28 terms
    
    # Let's build the 4x4x4 Strassen decomposition explicitly
    # The tensor has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # For the 4x4x4 case, we need rank-1 tensors
    
    # Actually, let me just implement the 4x4x4 Strassen and 4x4x2 schoolbook directly
    
    def strassen_4x4x4_terms():
        """Return list of (U_col, V_col, W_col) for 4x4x4 Strassen decomposition.
        U is 16x28, V is 16x28, W is 16x28"""
        terms = []
        
        # Define Strassen 2x2x2: for matrices with entries indexed by (r,c) in 0,1
        # A entries: a00,a01,a10,a11 -> positions in 4x4 block depend on which quadrant
        # B entries: b00,b01,b10,b11
        # The 7 products and their coefficients for C entries c00,c01,c10,c11
        
        # For a 2x2x2 block at position (bi, bj) where bi is row-block index, bj is col-block index
        # A block rows: 2*bi, 2*bi+1; A block cols: 2*bj, 2*bj+1
        # B block rows: 2*bk, 2*bk+1; B block cols: 2*bl, 2*bl+1
        # C block rows: 2*bi, 2*bi+1; C block cols: 2*bl, 2*bl+1
        
        # We have 4 2x2x2 multiplications:
        # (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2] ... wait no
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # So the four 2x2x2 multiplications are:
        # A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
        # That's 8 multiplications, but Strassen combines them in pairs to use 7 each? 
        # No, Strassen is for a single 2x2x2. For 4x4x4 with 2x2 blocks, we have 8 2x2x2 mults.
        # Each uses 7 terms = 56 terms total. That's worse than schoolbook 64? No, 56 < 64.
        
        # Wait, I'm confusing myself. The standard approach:
        # 4x4x4 can be done with 2x2x2 Strassen applied to each of the 4 blocks of the result,
        # but the inputs overlap. Actually, the standard result is that 4x4x4 matrix mult
        # can be done with rank 28 using a specific scheme, or rank 56 using 8 Strassen 2x2x2.
        
        # Let me use the simpler approach: 8 Strassen 2x2x2 blocks, each rank 7, total rank 56.
        # This is better than schoolbook 64.
        
        # For 4x4x4:
        # Block (bi, bj): A rows [2*bi, 2*bi+1], cols [2*bj, 2*bj+1]
        #                B rows [2*bj, 2*bj+1], cols [2*bj, 2*bj+1]  -- NO
        # C[bi,bj] = sum_k A[bi,k]*B[k,bj] where bi,bj,k in {0,1} are block indices
        # A block (bi,k): rows 2*bi..2*bi+1, cols 2*k..2*k+1
        # B block (k,bj): rows 2*k..2*k+1, cols 2*bj..2*bj+1
        # C block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        
        # So we have 8 block multiplications: (bi,k,bj) for bi,k,bj in {0,1}
        # Each is a 2x2x2 matrix multiplication.
        
        # Strassen 2x2x2: 
        # Input: A (2x2) with entries A[0,0],A[0,1],A[1,0],A[1,1]
        #        B (2x2) with entries B[0,0],B[0,1],B[1,0],B[1,1]
        # 7 products:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # Output:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # For the full 4x4x4, we need to express this in terms of rank-1 tensors.
        # Each Strassen product Mx corresponds to a rank-1 tensor in the 2x2x2 subspace.
        # We need to extend this to the full 4x4x4 tensor.
        
        # Let me define the 7 rank-1 tensors for a 2x2x2 block and then embed them.
        
        # For a 2x2x2 block at positions (a_row_offset, b_col_offset) in A, 
        # (b_row_offset, c_col_offset) in B, (a_row_offset, c_col_offset) in C:
        # The U vector has nonzeros at A positions, V at B positions, W at C positions.
        
        # Let's define the 7 Strassen rank-1 tensors in the 2x2x2 case:
        # Using the standard Strassen scheme, the 7 terms correspond to:
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        #   U: [1, 0, 0, 1] (A00, A01, A10, A11)
        #   V: [1, 0, 1, 0] (B00, B01, B10, B11)  
        #   Wait, B is indexed as B[j,k], so B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
        #   V for (B00+B11): [1, 0, 0, 1]
        #   W: coefficient in C. C00 += M1, C11 += M1
        #   W: [1, 0, 0, 1] (C00, C01, C10, C11)
        
        # Term 2: M2 = (A10+A11)*B00
        #   U: [0, 0, 1, 1]
        #   V: [1, 0, 0, 0]
        #   W: C10 += M2, C11 -= M2
        #   W: [0, 0, 1, -1]
        
        # Term 3: M3 = A00*(B01-B11)
        #   U: [1, 0, 0, 0]
        #   V: [0, 1, 0, -1]
        #   W: C01 += M3, C11 += M3
        #   W: [0, 1, 0, 1]
        
        # Term 4: M4 = A11*(B10-B00)
        #   U: [0, 0, 0, 1]
        #   V: [-1, 0, 1, 0]
        #   W: C00 += M4, C10 += M4
        #   W: [1, 0, 1, 0]
        
        # Term 5: M5 = (A00+A01)*B11
        #   U: [1, 1, 0, 0]
        #   V: [0, 0, 0, 1]
        #   W: C00 -= M5, C01 += M5
        #   W: [-1, 1, 0, 0]
        
        # Term 6: M6 = (A10-A00)*(B00+B01)
        #   U: [-1, 0, 1, 0]
        #   V: [1, 1, 0, 0]
        #   W: C11 += M6
        #   W: [0, 0, 0, 1]
        
        # Term 7: M7 = (A01-A11)*(B10+B11)
        #   U: [0, 1, 0, -1]
        #   V: [0, 0, 1, 1]
        #   W: C00 += M7
        #   W: [1, 0, 0, 0]
        
        # Let me verify C00 = M1 + M4 - M5 + M7
        # C00 from term 1: +1
        # C00 from term 4: +1
        # C00 from term 5: -1
        # C00 from term 7: +1
        # Sum: 1+1-1+1 = 2? That's wrong, should be 1.
        
        # I think I have an error. Let me re-derive Strassen.
        # Standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1+M4-M5+M7
        # c12 = M3+M5
        # c21 = M2+M4
        # c22 = M1-M2+M3+M6
        
        # Let me verify with a00=1, others 0 (A = identity top-left):
        # M1 = (1+0)(b11+b22) = b11+b22
        # M2 = (0+0)b11 = 0
        # M3 = 1*(b12-b22) = b12-b22
        # M4 = 0*(...) = 0
        # M5 = (1+0)b22 = b22
        # M6 = (0-1)(b11+b12) = -(b11+b12)
        # M7 = (0-0)(...) = 0
        # c11 = M1+M4-M5+M7 = (b11+b22) + 0 - b22 + 0 = b11 ✓ (should be a11*b11 + a12*b21 = b11)
        # c12 = M3+M5 = (b12-b22) + b22 = b12 ✓
        # c21 = M2+M4 = 0+0 = 0 ✓
        # c22 = M1-M2+M3+M6 = (b11+b22) - 0 + (b12-b22) + (-(b11+b12)) = b11+b22+b12-b22-b11-b12 = 0 ✓
        
        # Good. Now let me carefully build the rank-1 tensors.
        # A indexing: A[0,0]=a11, A[0,1]=a12, A[1,0]=a21, A[1,1]=a22
        # B indexing: B[0,0]=b11, B[0,1]=b12, B[1,0]=b21, B[1,1]=b22
        # C indexing: C[0,0]=c11, C[0,1]=c12, C[1,0]=c21, C[1,1]=c22
        
        # Term 1: M1 = (a11+a22)(b11+b22)
        #   U_A = [1, 0, 0, 1]
        #   U_B = [1, 0, 0, 1]
        #   U_C: c11 += 1, c22 += 1 => [1, 0, 0, 1]
        
        # Term 2: M2 = (a21+a22)*b11
        #   U_A = [0, 0, 1, 1]
        #   U_B = [1, 0, 0, 0]
        #   U_C: c21 += 1, c22 -= 1 => [0, 0, 1, -1]
        
        # Term 3: M3 = a11*(b12-b22)
        #   U_A = [1, 0, 0, 0]
        #   U_B = [0, 1, 0, -1]
        #   U_C: c12 += 1, c22 += 1 => [0, 1, 0, 1]
        
        # Term 4: M4 = a22*(b21-b11)
        #   U_A = [0, 0, 0, 1]
        #   U_B = [-1, 0, 1, 0]
        #   U_C: c11 += 1, c21 += 1 => [1, 0, 1, 0]
        
        # Term 5: M5 = (a11+a12)*b22
        #   U_A = [1, 1, 0, 0]
        #   U_B = [0, 0, 0, 1]
        #   U_C: c11 -= 1, c12 += 1 => [-1, 1, 0, 0]
        
        # Term 6: M6 = (a21-a11)*(b11+b12)
        #   U_A = [-1, 0, 1, 0]
        #   U_B = [1, 1, 0, 0]
        #   U_C: c22 += 1 => [0, 0, 0, 1]
        
        # Term 7: M7 = (a12-a22)*(b21+b22)
        #   U_A = [0, 1, 0, -1]
        #   U_B = [0, 0, 1, 1]
        #   U_C: c11 += 1 => [1, 0, 0, 0]
        
        strassen_2x2x2 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        # Now build 4x4x4 using 8 Strassen 2x2x2 blocks
        # Block (bi, bj): A rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
        #                B rows 2*bj:2*bj+2, cols 2*bj:2*bj+2  -- NO
        # Actually for C[bi,bj] = sum_k A[bi,k] * B[k,bj]:
        # A[bi,k]: rows 2*bi:2*bi+2, cols 2*k:2*k+2
        # B[k,bj]: rows 2*k:2*k+2, cols 2*bj:2*bj+2
        # C[bi,bj]: rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
        # So the blocks are indexed by (bi, k, bj) where bi,k,bj in {0,1}
        # There are 8 such blocks.
        
        U_444 = [[0] * 56 for _ in range(16)]
        V_444 = [[0] * 56 for _ in range(16)]
        W_444 = [[0] * 56 for _ in range(16)]
        
        col_idx = 0
        for bi in range(2):
            for k in range(2):
                for bj in range(2):
                    # A block: rows 2*bi to 2*bi+1, cols 2*k to 2*k+1
                    # B block: rows 2*k to 2*k+1, cols 2*bj to 2*bj+1
                    # C block: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
                    for t in range(7):
                        u_a = strassen_2x2x2[t][0]
                        u_b = strassen_2x2x2[t][1]
                        u_c = strassen_2x2x2[t][2]
                        
                        # Place in A: row 2*bi + r, col 2*k + c
                        for r in range(2):
                            for c in range(2):
                                U_444[(2*bi + r) * 4 + (2*k + c)][col_idx] = u_a[r * 2 + c]
                        
                        # Place in B: row 2*k + r, col 2*bj + c
                        for r in range(2):
                            for c in range(2):
                                V_444[(2*k + r) * 4 + (2*bj + c)][col_idx] = u_b[r * 2 + c]
                        
                        # Place in C: row 2*bi + r, col 2*bj + c
                        # W indexing: W[k*n + i] where k is col index of B, i is row index of A
                        # In our convention, W has shape (p*n, rank) = (4*4, 56)
                        # W[col * n + row]
                        for r in range(2):
                            for c in range(2):
                                W_444[(2*bj + c) * 4 + (2*bi + r)][col_idx] = u_c[r * 2 + c]
                        
                        col_idx += 1
        
        return U_444, V_444, W_444
    
    # For 4x4x2: schoolbook, rank = 4*4*2 = 32
    def schoolbook_4x4x2():
        U = [[0] * 32 for _ in range(16)]
        V = [[0] * 32 for _ in range(8)]
        W = [[0] * 32 for _ in range(8)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U[i * 4 + j][col] = 1
                    V[j * 2 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return U, V, W
    
    # Combine 4x4x4 (rank 56) and 4x4x2 (rank 32) for total rank 88
    # This is better than schoolbook 96.
    
    # But wait, we can also do 6 = 2 + 2 + 2, three 4x4x2 blocks = 96. Same as schoolbook.
    # Or 6 = 4 + 2: 56 + 32 = 88. Better.
    
    # Let's also try: can we do 4x4x3 better?
    # 4x4x3: schoolbook = 48. 
    # Split as 4x4x2 + 4x4x1 = 32 + 16 = 48. Same.
    # 4x4x1: just matrix-vector, rank 16.
    
    # So 4x4x6 = 4x4x4 + 4x4x2 = 56 + 32 = 88.
    
    # Let's verify the Strassen 4x4x4 is correct by checking the tensor identity.
    # For now, let's just construct it and hope it works.
    
    try:
        U_444, V_444, W_444 = strassen_4x4x4_terms()
        U_442, V_442, W_442 = schoolbook_4x4x2()
        
        # Combine: U is 16x88, V is 24x88, W is 24x88
        # Wait, V and W shapes:
        # V has shape (m*p, rank) = (4*6, rank) = (24, rank)
        # W has shape (p*n, rank) = (6*4, rank) = (24, rank)
        
        # For the 4x4x4 part: V_444 is 16x56 (only uses first 4 of 6 cols)
        # For the 4x4x2 part: V_442 is 8x32 (uses last 2 of 6 cols)
        # We need to embed these into 24-row vectors.
        
        rank_total = 56 + 32
        U_final = [[0] * rank_total for _ in range(16)]
        V_final = [[0] * rank_total for _ in range(24)]
        W_final = [[0] * rank_total for _ in range(24)]
        
        # Copy 4x4x4 part (cols 0-55)
        for i in range(16):
            for c in range(56):
                U_final[i][c] = U_444[i][c]
        for i in range(16):
            for c in range(56):
                V_final[i][c] = V_444[i][c]  # V rows 0-15 correspond to B cols 0-3
        for i in range(16):
            for c in range(56):
                W_final[i][c] = W_444[i][c]  # W rows 0-15 correspond to B cols 0-3
        
        # Copy 4x4x2 part (cols 56-87)
        for i in range(16):
            for c in range(32):
                U_final[i][56 + c] = U_442[i][c]
        # V_442 has 8 rows (B rows 0-3, B cols 0-1 of the 2-wide matrix)
        # These correspond to B rows 0-3, B cols 4-5 of the full 4x6 matrix
        for r in range(4):
            for c_local in range(2):
                for c in range(32):
                    V_final[r * 6 + 4 + c_local][56 + c] = V_442[r * 2 + c_local][c]
        # W_442 has 8 rows: W[k*n + i] for k in 0..1, i in 0..3
        # These correspond to B cols 4-5, A rows 0-3
        for k_local in range(2):
            for i in range(4):
                for c in range(32):
                    W_final[(4 + k_local) * 4 + i][56 + c] = W_442[k_local * 4 + i][c]
        
        # Verify by checking tensor identity on random inputs
        if time.time() - start_time < deadline - 30:
            np.random.seed(42)
            A = np.random.randn(4, 4)
            B = np.random.randn(4, 6)
            C_expected = A @ B
            
            U_arr = np.array(U_final)
            V_arr = np.array(V_final)
            W_arr = np.array(W_final)
            
            # C = U^T * V * W^T? No.
            # The tensor T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # where i is A row+col index, j is B row+col index, k is C row+col index
            # C[row, col] = sum_j A[row, j] * B[j, col]
            # So T should satisfy: sum_c U[row*4+j, c] * V[j*6+col, c] * W[col*4+row, c] = C[row, col]
            
            C_computed = np.zeros((4, 6))
            for row in range(4):
                for col in range(6):
                    s = 0
                    for j in range(4):
                        for c in range(rank_total):
                            s += U_arr[row * 4 + j, c] * V_arr[j * 6 + col, c] * W_arr[col * 4 + row, c]
                    C_computed[row, col] = s
            
            if np.allclose(C_computed, C_expected, atol=1e-10):
                return {"U": U_final, "V": V_final, "W": W_final}
            else:
                # Strassen embedding is wrong, fall back to schoolbook
                return schoolbook()
        
        return schoolbook()
        
    except Exception:
        return schoolbook()

# EVOLVE-BLOCK-END
