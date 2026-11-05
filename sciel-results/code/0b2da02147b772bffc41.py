import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will partition the 3x5x6 multiplication into:
    # 1. A 2x4x5 sub-problem (rows 0-1, cols 0-3 of A, rows 0-3 of B, cols 0-4 of C)
    # 2. The remaining terms which form a structure that can be handled with rank 3.
    # Total rank = 17 (for 2x4x5) + 3 = 20.
    # Wait, let's verify the partition.
    # A is 3x5, B is 5x6, C is 3x6.
    # Let's split A's columns into 4 and 1.
    # A = [A1 (3x4) | A2 (3x1)]
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6.
    # C = A1 B1 + A2 B2.
    
    # Term 1: A1 (3x4) * B1 (4x6) -> C1 (3x6)
    # We only need the top 2 rows of C1 to be part of the "2x4x5" block?
    # No, the directive says "partition ... into a 2x4x5 block".
    # This implies we are only decomposing a sub-tensor of the full multiplication tensor.
    # But the output C must be the full 3x6 matrix.
    # So we must decompose the FULL tensor.
    
    # Let's look at the structure:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # If we use the 2x4x5 rank 17 decomposition, it computes C_sub[i,k] for i in 0..1, k in 0..4.
    # This covers 2 rows of C and 5 columns of C.
    # The remaining parts of C are:
    # - Row 2 of C, all 6 columns.
    # - Rows 0-1, column 5 of C.
    
    # Let's define the indices:
    # A: 3x5 (i=0..2, j=0..4)
    # B: 5x6 (j=0..4, k=0..5)
    # C: 3x6 (i=0..2, k=0..5)
    
    # Block 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # This is a 2x4x5 sub-tensor.
    # The rank of this sub-tensor is at most 17 (known result for 2x4x5? Actually R(2,4,5) is not 17, R(2,2,2)=7. R(2,4,4) = 2*R(2,2,4)? No.
    # Let's check known ranks.
    # R(2,4,5) <= R(2,4,4) + R(2,4,1) = R(2,4,4) + 8.
    # R(2,4,4) <= 2*R(2,2,4)? No.
    # Actually, the directive claims "proven rank-17 decomposition for the 2x4x5 tensor". I will trust this premise for the sake of the "research programme" directive, but I need to implement it.
    # If I don't know the coefficients, I can't implement it.
    
    # Alternative: Use the known R(2,2,2)=7.
    # Can I decompose 3x5x6 using 2x2x2 Strassen blocks?
    # Pad to 4x6x6?
    # R(4,6,6) <= 2*R(2,6,6) + 2*R(2,6,6)? No.
    # R(4,6,6) <= 2*R(2,3,6) + 2*R(2,3,6)?
    
    # Let's try a simpler partition that I can implement with known Strassen coefficients.
    # R(3,5,6) <= R(2,5,6) + R(1,5,6)
    # R(1,5,6) = 6 (schoolbook, rank 1x5x6 is 6? No, rank 1 for each output? No.
    # R(1,m,p) = m*p? No, R(1,m,p) = m*p is schoolbook.
    # Actually R(1,m,p) = m*p is not minimal. R(1,1,1)=1.
    # For 1x5x6, we compute C[0,k] = sum_j A[0,j] B[j,k].
    # This is 6 independent dot products of length 5.
    # Can we do better?
    # R(1,5,6) <= 5*6 = 30.
    # Is there a lower rank?
    # R(1,m,p) = m*p is the schoolbook.
    # Actually, R(1,m,p) is not well-studied for small m,p in terms of Strassen-like improvements because one dimension is 1.
    # Wait, R(1,2,2) = 4? C[0,0]=A0B0+A1B1, C[0,1]=A0B2+A1B3.
    # U: 1x2, V: 2x2, W: 2x1.
    # Terms:
    # t1 = A0 B0 -> C00
    # t2 = A1 B1 -> C00
    # t3 = A0 B2 -> C01
    # t4 = A1 B3 -> C01
    # Rank 4.
    # Can we do rank 3?
    # If we use Strassen on 2x2? No, one dim is 1.
    # So R(1,5,6) = 30 is likely optimal or close.
    
    # So R(3,5,6) <= R(2,5,6) + 30.
    # R(2,5,6) <= R(2,4,6) + R(2,1,6) = R(2,4,6) + 12.
    # R(2,4,6) <= R(2,4,4) + R(2,4,2) = R(2,4,4) + 16?
    # R(2,4,4) <= 2*R(2,2,4)?
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14?
    # So R(2,4,4) <= 28.
    # R(2,4,6) <= 28 + 16 = 44.
    # R(2,5,6) <= 44 + 12 = 56.
    # Total R(3,5,6) <= 56 + 30 = 86.
    # Schoolbook is 90.
    # 86 is an improvement.
    
    # Let's verify R(2,2,4) = 14.
    # A(2x2) * B(2x4) -> C(2x4).
    # Split B into B1(2x2) and B2(2x2).
    # C = A B1 + A B2.
    # Use Strassen for A B1 (rank 7) and Strassen for A B2 (rank 7).
    # Total rank 14.
    # Yes, this works.
    
    # Let's verify R(2,4,2) = 16.
    # A(2x4) * B(4x2) -> C(2x2).
    # Split A into A1(2x2) and A2(2x2).
    # C = A1 B1 + A2 B2 where B1, B2 are 2x2 (cols of B).
    # Wait, B is 4x2. B1 is top 2 rows, B2 is bottom 2 rows.
    # C = A1 B1 + A2 B2.
    # A1 is 2x2, B1 is 2x2. Product is 2x2.
    # A2 is 2x2, B2 is 2x2. Product is 2x2.
    # We can use Strassen for A1 B1 (rank 7) and Strassen for A2 B2 (rank 7).
    # Total rank 14?
    # But the outputs add up. C = C1 + C2.
    # The decomposition of A1 B1 has 7 terms.
    # The decomposition of A2 B2 has 7 terms.
    # We just add the corresponding W terms?
    # Yes, because the tensor is linear.
    # So R(2,4,2) <= 14.
    
    # So R(2,4,6) <= 28 + 14 = 42.
    # R(2,5,6) <= 42 + 12 = 54.
    # R(3,5,6) <= 54 + 30 = 84.
    
    # Let's refine R(1,5,6).
    # Is R(1,5,6) really 30?
    # C[0,k] = sum_j A[0,j] B[j,k].
    # This is 6 independent linear forms in A[0,j] for each k.
    # Actually, for each k, it's a dot product.
    # Can we share multiplications?
    # No, because A[0,j] appears in all k, but B[j,k] is different.
    # It's a bilinear form.
    # R(1,m,p) = m*p is the standard schoolbook.
    # Is there a Strassen-like trick?
    # For 1x2x2, rank 4.
    # For 1x2x3, rank 6.
    # It seems R(1,m,p) = m*p.
    
    # So 84 is a valid bound.
    # I need to implement this.
    # The implementation requires:
    # 1. Decompose 2x2x2 (Strassen, rank 7).
    # 2. Combine to 2x2x4 (rank 14).
    # 3. Combine to 2x4x4 (rank 28).
    # 4. Combine to 2x4x2 (rank 14).
    # 5. Combine to 2x4x6 (rank 42).
    # 6. Combine to 2x5x6 (rank 54).
    # 7. Add 1x5x6 (rank 30).
    # Total 84.
    
    # Let's write the Strassen 2x2x2 decomposition.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # I will construct the U, V, W matrices for the full 3x5x6 problem.
    # U shape: (3*5, rank) = (15, 84)
    # V shape: (5*6, rank) = (30, 84)
    # W shape: (6*3, rank) = (18, 84)
    
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # We will build the rank 84 decomposition.
    # Rank 0-7: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Rank 8-14: Strassen for A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Wait, this is part of 2x4x2)
    # Let's be systematic.
    
    # Part 1: 2x4x6 (rows 0-1 of A, cols 0-5 of C)
    # Split into 2x4x4 (cols 0-3) and 2x4x2 (cols 4-5).
    
    # 2x4x4 (cols 0-3):
    # Split into 2x2x4 (cols 0-1) and 2x2x4 (cols 2-3)? No, 2x4x4 means A is 2x4.
    # A is 2x4. B is 4x4. C is 2x4.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B11, B12 (top 2 rows) and B21, B22 (bottom 2 rows).
    # C = A1 [B11 B12] + A2 [B21 B22]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # So we need:
    # 1. 2x2x2: A1 B11 -> C11 (rank 7)
    # 2. 2x2x2: A1 B12 -> C12 (rank 7)
    # 3. 2x2x2: A2 B21 -> C21 (rank 7)
    # 4. 2x2x2: A2 B22 -> C22 (rank 7)
    # Total rank 28 for 2x4x4.
    
    # 2x4x2 (cols 4-5):
    # A is 2x4. B is 4x2. C is 2x2.
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B into B1 (2x2, top rows) and B2 (2x2, bottom rows).
    # C = A1 B1 + A2 B2.
    # 1. 2x2x2: A1 B1 -> C (rank 7)
    # 2. 2x2x2: A2 B2 -> C (rank 7)
    # Total rank 14 for 2x4x2.
    
    # So 2x4x6 rank is 28 + 14 = 42.
    
    # Part 2: 1x5x6 (row 2 of A, cols 0-5 of C)
    # A is 1x5. B is 5x6. C is 1x6.
    # Schoolbook rank 30.
    # For each k in 0..5, C[0,k] = sum_j A[0,j] B[j,k].
    # This is 6 terms? No, 5*6=30 terms.
    # Each term is A[0,j] * B[j,k].
    # U[0*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + 0, r] = 1
    # There are 30 such terms.
    
    # Total rank = 42 + 30 = 72.
    # Wait, earlier I calculated 84. Why?
    # R(2,5,6) <= R(2,4,6) + R(2,1,6).
    # R(2,1,6) = 12.
    # R(2,4,6) = 42.
    # R(2,5,6) = 54.
    # R(3,5,6) = R(2,5,6) + R(1,5,6) = 54 + 30 = 84.
    
    # Why did I get 72 above?
    # I decomposed 2x4x6 (rows 0-1, all 6 cols).
    # But A is 3x5.
    # Rows 0-1 of A are 2x5.
    # I used 2x4x6. This uses cols 0-3 of A.
    # What about col 4 of A (rows 0-1)?
    # That is a 2x1x6 sub-problem.
    # R(2,1,6) = 12.
    # So R(2,5,6) = R(2,4,6) + R(2,1,6) = 42 + 12 = 54.
    # Then R(3,5,6) = R(2,5,6) + R(1,5,6) = 54 + 30 = 84.
    
    # So total rank is 84.
    
    # Let's implement this.
    
    rank = 84
    U = [[0] * rank for _ in range(3 * 5)]
    V = [[0] * rank for _ in range(5 * 6)]
    W = [[0] * rank for _ in range(6 * 3)]
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, rank_offset):
        # Adds a 2x2x2 Strassen decomposition.
        # A is at a_row_offset, a_col_offset (2x2)
        # B is at b_row_offset, b_col_offset (2x2)
        # C is at c_row_offset, c_col_offset (2x2)
        # n=3, m=5, p=6
        
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 terms.
        # Term 0: M1
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00, C11 -> 1
        
        # Term 1: M2
        # U: A10, A11 -> 1
        # V: B00 -> 1
        # W: C10 -> 1
        
        # Term 2: M3
        # U: A00 -> 1
        # V: B01 -> 1, B11 -> -1
        # W: C01 -> 1
        
        # Term 3: M4
        # U: A11 -> 1
        # V: B10 -> 1, B00 -> -1
        # W: C00 -> 1, C10 -> 1
        
        # Term 4: M5
        # U: A00, A01 -> 1
        # V: B11 -> 1
        # W: C00 -> -1, C01 -> 1
        
        # Term 5: M6
        # U: A10 -> 1, A00 -> -1
        # V: B00, B01 -> 1
        # W: C11 -> 1
        
        # Term 6: M7
        # U: A01 -> 1, A11 -> -1
        # V: B10, B11 -> 1
        # W: C00 -> 1, C11 -> -1
        
        # Helper to set U, V, W
        def set_u(row, col, val):
            U[row][rank_offset] += val
            
        def set_v(row, col, val):
            V[row][rank_offset] += val
            
        def set_w(row, col, val):
            W[row][rank_offset] += val
        
        # Indices
        # A[i,j] -> U[a_row_offset + i, a_col_offset + j]
        # B[i,j] -> V[b_row_offset + i, b_col_offset + j]
        # C[i,j] -> W[c_col_offset + j, c_row_offset + i]  # Note: W[k*n + i, r]
        
        # Term 0: M1
        set_u(a_row_offset + 0, a_col_offset + 0, 1)
        set_u(a_row_offset + 1, a_col_offset + 1, 1)
        set_v(b_row_offset + 0, b_col_offset + 0, 1)
        set_v(b_row_offset + 1, b_col_offset + 1, 1)
        set_w(c_col_offset + 0, c_row_offset + 0, 1) # C00
        set_w(c_col_offset + 1, c_row_offset + 1, 1) # C11
        rank_offset += 1
        
        # Term 1: M2
        set_u(a_row_offset + 1, a_col_offset + 0, 1)
        set_u(a_row_offset + 1, a_col_offset + 1, 1)
        set_v(b_row_offset + 0, b_col_offset + 0, 1)
        set_w(c_col_offset + 0, c_row_offset + 1, 1) # C10
        rank_offset += 1
        
        # Term 2: M3
        set_u(a_row_offset + 0, a_col_offset + 0, 1)
        set_v(b_row_offset + 0, b_col_offset + 1, 1)
        set_v(b_row_offset + 1, b_col_offset + 1, -1)
        set_w(c_col_offset + 1, c_row_offset + 0, 1) # C01
        rank_offset += 1
        
        # Term 3: M4
        set_u(a_row_offset + 1, a_col_offset + 1, 1)
        set_v(b_row_offset + 1, b_col_offset + 0, 1)
        set_v(b_row_offset + 0, b_col_offset + 0, -1)
        set_w(c_col_offset + 0, c_row_offset + 0, 1) # C00
        set_w(c_col_offset + 0, c_row_offset + 1, 1) # C10
        rank_offset += 1
        
        # Term 4: M5
        set_u(a_row_offset + 0, a_col_offset + 0, 1)
        set_u(a_row_offset + 0, a_col_offset + 1, 1)
        set_v(b_row_offset + 1, b_col_offset + 1, 1)
        set_w(c_col_offset + 0, c_row_offset + 0, -1) # C00
        set_w(c_col_offset + 1, c_row_offset + 0, 1) # C01
        rank_offset += 1
        
        # Term 5: M6
        set_u(a_row_offset + 1, a_col_offset + 0, 1)
        set_u(a_row_offset + 0, a_col_offset + 0, -1)
        set_v(b_row_offset + 0, b_col_offset + 0, 1)
        set_v(b_row_offset + 0, b_col_offset + 1, 1)
        set_w(c_col_offset + 1, c_row_offset + 1, 1) # C11
        rank_offset += 1
        
        # Term 6: M7
        set_u(a_row_offset + 0, a_col_offset + 1, 1)
        set_u(a_row_offset + 1, a_col_offset + 1, -1)
        set_v(b_row_offset + 1, b_col_offset + 0, 1)
        set_v(b_row_offset + 1, b_col_offset + 1, 1)
        set_w(c_col_offset + 0, c_row_offset + 0, 1) # C00
        set_w(c_col_offset + 1, c_row_offset + 1, -1) # C11
        rank_offset += 1
        
        return rank_offset

    # Part 1: 2x4x6 (Rows 0-1 of A, Cols 0-5 of C)
    # A is 2x4 (rows 0-1, cols 0-3)
    # B is 4x6 (rows 0-3, cols 0-5)
    # C is 2x6 (rows 0-1, cols 0-5)
    
    # 1a: 2x2x4 (A rows 0-1, cols 0-1; B rows 0-1, cols 0-3; C rows 0-1, cols 0-3)
    # Split into two 2x2x2.
    # 1a1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    rank_offset = add_strassen(0, 0, 0, 0, 0, 0, 0)
    # 1a2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    rank_offset = add_strassen(0, 0, 0, 2, 0, 2, rank_offset)
    
    # 1b: 2x2x4 (A rows 0-1, cols 2-3; B rows 2-3, cols 0-3; C rows 0-1, cols 0-3)
    # 1b1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    rank_offset = add_strassen(0, 2, 2, 0, 0, 0, rank_offset)
    # 1b2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    rank_offset = add_strassen(0, 2, 2, 2, 0, 2, rank_offset)
    
    # 1c: 2x4x2 (A rows 0-1, cols 0-3; B rows 0-3, cols 4-5; C rows 0-1, cols 4-5)
    # Split into two 2x2x2.
    # 1c1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    rank_offset = add_strassen(0, 0, 0, 4, 0, 4, rank_offset)
    # 1c2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    rank_offset = add_strassen(0, 2, 2, 4, 0, 4, rank_offset)
    
    # Part 2: 2x1x6 (Rows 0-1 of A, Col 4 of A; Cols 0-5 of C)
    # A is 2x1 (rows 0-1, col 4)
    # B is 1x6 (row 4, cols 0-5)
    # C is 2x6 (rows 0-1, cols 0-5)
    # Schoolbook rank 12.
    # C[i,k] = A[i,4] * B[4,k]
    for i in range(2):
        for k in range(6):
            U[i * 5 + 4][rank_offset] = 1
            V[4 * 6 + k][rank_offset] = 1
            W[k * 3 + i][rank_offset] = 1
            rank_offset += 1
            
    # Part 3: 1x5x6 (Row 2 of A, Cols 0-5 of C)
    # A is 1x5 (row 2, cols 0-4)
    # B is 5x6 (rows 0-4, cols 0-5)
    # C is 1x6 (row 2, cols 0-5)
    # Schoolbook rank 30.
    # C[2,k] = sum_j A[2,j] B[j,k]
    for j in range(5):
        for k in range(6):
            U[2 * 5 + j][rank_offset] = 1
            V[j * 6 + k][rank_offset] = 1
            W[k * 3 + 2][rank_offset] = 1
            rank_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
