"""Matrix multiplication 2,4,7 — rank 50 decomposition via Strassen + schoolbook blocks."""
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 120.0

    n, m, p = 2, 4, 7

    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01+B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each M_r is a rank-1 tensor.
    # Local indices: A[i,j] with i in {0,1}, j in {0,1}; B[j,k] with j in {0,1}, k in {0,1}; C[i,k]
    # For rank-1: U_vec[i*m+j], V_vec[j*p+k], W_vec[k*n+i]
    # But locally for 2x2x2: U_local[i*2+j], V_local[j*2+k], W_local[k*2+i]

    # Coefficients for A part (U): dict from (i,j) to coeff
    # Coefficients for B part (V): dict from (j,k) to coeff
    # Coefficients for C part (W): dict from (i,k) to coeff

    def strassen_rank1_tensors():
        """Return list of (U_coeffs, V_coeffs, W_coeffs) for the 7 rank-1 tensors of Strassen 2x2x2."""
        tensors = []
        # M1 = (A00+A11)(B00+B11)
        U1 = {(0,0): 1, (1,1): 1}
        V1 = {(0,0): 1, (1,1): 1}
        W1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}  # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, I need to be careful. The rank-1 decomposition means:
        # C[i,k] = sum_r U[i,j] * A[i,j] * V[j,k] * B[j,k] ... 
        # Actually no. The tensor T[i,j,k] = delta (the multiplication tensor) satisfies:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor has entries T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to C[i,k]
        # i.e., T[i,j,k] = 1 for all valid i,j,k where j is shared.
        
        # In rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector over (i,j) pairs, v_r over (j,k) pairs, w_r over (i,k) pairs.
        # And T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
        
        # For Strassen, each M_r = (linear combo of A) * (linear combo of B) contributes to certain C entries.
        # M_r = (sum_{i,j} a^{r}_{i,j} A[i,j]) * (sum_{j,k} b^{r}_{j,k} B[j,k])
        # And it contributes to C[i,k] with coefficient c^{r}_{i,k}.
        
        # So u_r[i,j] = a^{r}_{i,j}, v_r[j,k] = b^{r}_{j,k}, w_r[i,k] = c^{r}_{i,k}
        
        # Let me re-derive:
        # M1 = (A00+A11)(B00+B11)
        # a^1: (0,0):1, (1,1):1
        # b^1: (0,0):1, (1,1):1
        # M1 contributes to C00, C01, C10, C11? No!
        # M1 is a scalar. C00 gets +M1, C11 gets +M1. C01 and C10 don't directly get M1.
        
        # Let me redo:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M1: contributes to C00 (+1), C11 (+1). Not to C01, C10.
        # M2: contributes to C10 (+1), C11 (-1).
        # M3: contributes to C01 (+1), C11 (+1).
        # M4: contributes to C00 (+1), C10 (+1).
        # M5: contributes to C00 (-1), C01 (+1).
        # M6: contributes to C11 (+1).
        # M7: contributes to C00 (+1).
        
        # Hmm wait, let me double-check Strassen:
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Yes that's what I have. Let me verify M1:
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This contributes to C00 (A00B00), C11 (A11B11), C01 (A00B11? no, A00B11 is in C00's row... )
        
        # Wait I'm confusing myself. Let me think more carefully.
        # A is 2x2: A[i][j], i=row, j=col
        # B is 2x2: B[j][k], j=row, k=col
        # C is 2x2: C[i][k] = sum_j A[i][j] B[j][k]
        
        # So C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
        # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
        # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
        # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01+B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let me verify C00:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B10 - A11B00
        # -M5 = -A00B11 - A01B11
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        # Sum: A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + (A00B11 - A00B11) + (A11B00 - A11B00) + (A11B11 - A11B11) + (A11B10 - A11B10) + (-A01B11 + A01B11) + A01B10
        # = A00B00 + A01B10
        # But C00 = A00B00 + A01B10. Yes! Correct.
        
        # So the coefficients are:
        tensors = []
        # M1
        U1 = {(0,0): 1, (1,1): 1}
        V1 = {(0,0): 1, (1,1): 1}
        W1 = {(0,0): 1, (1,1): 1}
        tensors.append((U1, V1, W1))
        # M2
        U2 = {(1,0): 1, (1,1): 1}
        V2 = {(0,0): 1}
        W2 = {(1,0): 1, (1,1): -1}
        tensors.append((U2, V2, W2))
        # M3
        U3 = {(0,0): 1}
        V3 = {(0,1): 1, (1,1): 1}
        W3 = {(0,1): 1, (1,1): 1}
        tensors.append((U3, V3, W3))
        # M4
        U4 = {(1,1): 1}
        V4 = {(1,0): 1, (0,0): -1}
        W4 = {(0,0): 1, (1,0): 1}
        tensors.append((U4, V4, W4))
        # M5
        U5 = {(0,0): 1, (0,1): 1}
        V5 = {(1,1): 1}
        W5 = {(0,0): -1, (0,1): 1}
        tensors.append((U5, V5, W5))
        # M6
        U6 = {(1,0): 1, (0,0): -1}
        V6 = {(0,0): 1, (0,1): 1}
        W6 = {(1,1): 1}
        tensors.append((U6, V6, W6))
        # M7
        U7 = {(0,1): 1, (1,1): -1}
        V7 = {(1,0): 1, (1,1): 1}
        W7 = {(0,0): 1}
        tensors.append((U7, V7, W7))
        
        return tensors

    def schoolbook_rank1_tensors():
        """Return list of (U_coeffs, V_coeffs, W_coeffs) for the 4 rank-1 tensors of schoolbook 2x2x2."""
        tensors = []
        for i in range(2):
            for j in range(2):
                U = {(i,j): 1}
                for k in range(2):
                    V = {(j,k): 1}
                    W = {(i,k): 1}
                    tensors.append((U, V, W))
        return tensors

    # Now build the global decomposition.
    # n=2, m=4, p=7
    # A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can view this as: for each j (0..3), A[:,j] (2x1) times B[j,:] (1x7) contributes to C.
    # Alternatively, group columns of A and rows of B.
    
    # The approach: partition the 4 middle dimension into blocks.
    # We have 4 columns in A and 4 rows in B.
    # We can group them as: two groups of 2 for Strassen (covering 4 middle indices),
    # but wait, Strassen works on 2x2 blocks. The middle dimension is 4.
    
    # Actually, let me think differently. The tensor for 2x4x7 matmul has rank at most 2*4*7=56.
    # We want rank 50.
    
    # One approach: use the fact that we can do the 4x7 part with some structure.
    # 
    # Let's think of it as: A is 2x4, B is 4x7.
    # We can write B as a 4x7 matrix. 
    # 
    # Alternative: use the identity that R(n,m,p) <= R(n,m,p1) + R(n,m,p2) if p=p1+p2.
    # So R(2,4,7) <= R(2,4,4) + R(2,4,3).
    # R(2,4,4): A is 2x4, B is 4x4, C is 2x4.
    # R(2,4,4) <= R(2,2,4) + R(2,2,4) = 2*R(2,2,4).
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 2*7 = 14.
    # So R(2,4,4) <= 28.
    # R(2,4,3) <= R(2,2,3) + R(2,2,3) = 2*R(2,2,3).
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # So R(2,4,3) <= 22.
    # Total: 28 + 22 = 50. 
    #
    # This matches! So the construction is:
    # Split the 7 output columns into 4 and 3.
    # For the 4-column part: split the 4 input columns (middle) into two groups of 2, apply Strassen to each 2x2x2 block.
    # Wait, that's not quite right either.
    
    # Let me reconsider. R(2,4,7) <= R(2,4,4) + R(2,4,3).
    # For R(2,4,4): A is 2x4, B is 4x4.
    # We can further decompose: R(2,4,4) <= R(2,2,4) + R(2,2,4) where we split the 4 columns of A into two groups of 2.
    # For each R(2,2,4): A is 2x2, B is 2x4. R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14.
    # So each R(2,2,4) uses two Strassen blocks.
    # Total for R(2,4,4): 2*14 = 28.
    
    # For R(2,4,3): A is 2x4, B is 4x3.
    # R(2,4,3) <= R(2,2,3) + R(2,2,3) = 2*R(2,2,3).
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # So each R(2,2,3) uses one Strassen (2x2x2) and one schoolbook (2x2x1).
    # Total for R(2,4,3): 2*11 = 22.
    
    # Grand total: 28 + 22 = 50.
    
    # Now let's implement this carefully.
    
    # Global indices:
    # A[i,j]: i in 0..1, j in 0..3
    # B[j,k]: j in 0..3, k in 0..6
    # C[i,k]: i in 0..1, k in 0..6
    
    # U vector: index (i*4+j), shape 8
    # V vector: index (j*7+k), shape 28
    # W vector: index (k*2+i), shape 14
    
    # Step 1: Split columns of C into k=0..3 and k=4..6.
    # Step 2: For k=0..3 (the 4-column part):
    #   Split rows of B (and columns of A) into j=0,1 and j=2,3.
    #   For j=0,1: A[:,0:2] is 2x2, B[0:2,0:4] is 2x4.
    #     This is R(2,2,4) = R(2,2,2) + R(2,2,2).
    #     Split B[0:2,0:4] into B[0:2,0:2] and B[0:2,2:4], each 2x2.
    #     Apply Strassen to each.
    #   For j=2,3: A[:,2:4] is 2x2, B[2:4,0:4] is 2x4.
    #     Similarly apply two Strassen blocks.
    #   Total: 4 Strassen blocks = 4*7 = 28.
    
    # Step 3: For k=4..6 (the 3-column part):
    #   Split rows of B (and columns of A) into j=0,1 and j=2,3.
    #   For j=0,1: A[:,0:2] is 2x2, B[0:2,4:7] is 2x3.
    #     This is R(2,2,3) = R(2,2,2) + R(2,2,1).
    #     Split B[0:2,4:7] into B[0:2,4:6] (2x2) and B[0:2,6:7] (2x1).
    #     Apply Strassen to the 2x2 part, schoolbook to the 2x1 part.
    #   For j=2,3: A[:,2:4] is 2x2, B[2:4,4:7] is 2x3.
    #     Similarly.
    #   Total: 2*(7+4) = 22.
    
    # Grand total: 28 + 22 = 50.
    
    U = [0] * (8 * 50)
    V = [0] * (28 * 50)
    W = [0] * (14 * 50)
    
    rank = 50
    col = 0
    
    strassen_tensors = strassen_rank1_tensors()
    schoolbook_tensors = schoolbook_rank1_tensors()
    
    def add_strassen_block(a_col_offset, b_row_offset, b_col_offset):
        """Add a Strassen block where:
        A uses columns [a_col_offset, a_col_offset+1]
        B uses rows [b_row_offset, b_row_offset+1] and columns [b_col_offset, b_col_offset+1]
        C uses rows 0..1 and columns [b_col_offset, b_col_offset+1]
        """
        nonlocal col
        for (u_coeff, v_coeff, w_coeff) in strassen_tensors:
            for (li, lj), cu in u_coeff.items():
                gi = 0  # A row is always 0..1, but li is local row
                gj = a_col_offset + lj
                U[gi * 4 + gj] += cu * (1 if col >= 0 else 0)  # This is wrong, let me fix
            # I need to be more careful. Let me rewrite.
        pass

    # Let me restart the construction more carefully.
    
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(14)]
    
    col = 0
    
    def add_strassen(a_col_off, b_row_off, b_col_off):
        """Add Strassen block for A[:, a_col_off:a_col_off+2], B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]"""
        nonlocal col
        for (u_coeff, v_coeff, w_coeff) in strassen_tensors:
            # u_coeff maps (local_i, local_j) -> coeff
            # local_i is row of A (0 or 1), local_j is col of A within the 2-block
            # Global A index: row=local_i, col=a_col_off+local_j
            # Global U index: local_i * 4 + (a_col_off + local_j)
            # v_coeff maps (local_j, local_k) -> coeff
            # local_j is row of B (0 or 1), local_k is col of B within the 2-block
            # Global B index: row=b_row_off+local_j, col=b_col_off+local_k
            # Global V index: (b_row_off+local_j) * 7 + (b_col_off+local_k)
            # w_coeff maps (local_i, local_k) -> coeff
            # Global C index: row=local_i, col=b_col_off+local_k
            # Global W index: (b_col_off+local_k) * 2 + local_i
            for (li, lj), cu in u_coeff.items():
                U[li * 4 + (a_col_off + lj)][col] += cu
            for (lj, lk), cv in v_coeff.items():
                V[(b_row_off + lj) * 7 + (b_col_off + lk)][col] += cv
            for (li, lk), cw in w_coeff.items():
                W[(b_col_off + lk) * 2 + li][col] += cw
            col += 1
    
    def add_schoolbook(a_col_off, b_row_off, b_col_off):
        """Add schoolbook block for A[:, a_col_off:a_col_off+1] (single col), B[b_row_off, b_col_off] (single entry)
        Actually for 2x2x1: A is 2x1 (one column), B is 1x1... no.
        2x2x1: A is 2x2? No. Let me reconsider.
        R(2,2,1): A is 2x2, B is 2x1, C is 2x1.
        Wait, that doesn't make sense for matmul. A(2x2) * B(2x1) = C(2x1). Yes.
        But in our context, when we say "schoolbook 2x2x1", we mean:
        A is 2x2 (two columns of the original A), B is 2x1 (one row of B for one output col), C is 2x1.
        Actually no. Let me re-examine.
        
        When I said R(2,2,3) = R(2,2,2) + R(2,2,1), I meant:
        A is 2x2, B is 2x3, C is 2x3.
        Split B into B[:, 0:2] (2x2) and B[:, 2:3] (2x1).
        R(2,2,2) for the 2x2 part.
        R(2,2,1) for the 2x1 part.
        
        For R(2,2,1): A is 2x2, B is 2x1, C is 2x1.
        C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        This has rank 4 (schoolbook): 4 rank-1 terms, one for each (i,j) pair... wait.
        C[i,0] = sum_j A[i,j] B[j,0]
        The tensor T[i,j,k] where k is always 0: T[i,j,0] = 1.
        Rank of this tensor: it's the tensor for 2x2 x 1 matmul.
        R(2,2,1) = 4. Yes, schoolbook.
        
        So for R(2,2,1): A is 2x2, B is 2x1 (one column), C is 2x1.
        The 4 schoolbook terms are for (i,j) in {(0,0),(0,1),(1,0),(1,1)}:
        C[i,0] += A[i,j] * B[j,0]
        """
        nonlocal col
        # A is 2x2: columns a_col_off, a_col_off+1
        # B is 2x1: row b_row_off, b_row_off+1, column b_col_off
        # C is 2x1: rows 0,1, column b_col_off
        for i in range(2):
            for j in range(2):
                # U: A[i, a_col_off+j]
                U[i * 4 + (a_col_off + j)][col] = 1
                # V: B[b_row_off+j, b_col_off]
                V[(b_row_off + j) * 7 + b_col_off][col] = 1
                # W: C[i, b_col_off]
                W[b_col_off * 2 + i][col] = 1
                col += 1
    
    # Now build the full decomposition:
    
    # Part 1: k=0..3 (4 output columns), split into j=0,1 and j=2,3
    # For j=0,1 (a_col_off=0, b_row_off=0):
    #   B columns 0,1: add_strassen(0, 0, 0)
    #   B columns 2,3: add_strassen(0, 0, 2)
    # For j=2,3 (a_col_off=2, b_row_off=2):
    #   B columns 0,1: add_strassen(2, 2, 0)
    #   B columns 2,3: add_strassen(2, 2, 2)
    
    add_strassen(0, 0, 0)   # 7 terms, col 0-6
    add_strassen(0, 0, 2)   # 7 terms, col 7-13
    add_strassen(2, 2, 0)   # 7 terms, col 14-20
    add_strassen(2, 2, 2)   # 7 terms, col 21-27
    
    # Part 2: k=4..6 (3 output columns), split into j=0,1 and j=2,3
    # For j=0,1 (a_col_off=0, b_row_off=0):
    #   B columns 4,5 (2x2): add_strassen(0, 0, 4)
    #   B column 6 (2x1): add_schoolbook(0, 0, 6)
    # For j=2,3 (a_col_off=2, b_row_off=2):
    #   B columns 4,5 (2x2): add_strassen(2, 2, 4)
    #   B column 6 (2x1): add_schoolbook(2, 2, 6)
    
    add_strassen(0, 0, 4)   # 7 terms, col 28-34
    add_schoolbook(0, 0, 6) # 4 terms, col 35-38
    add_strassen(2, 2, 4)   # 7 terms, col 39-45
    add_schoolbook(2, 2, 6) # 4 terms, col 46-49
    
    # col should be 50 now
    assert col == 50, f"Expected col=50, got {col}"
    
    # Convert to flat lists
    U_flat = [U[i][c] for i in range(8) for c in range(50)]
    V_flat = [V[i][c] for i in range(28) for c in range(50)]
    W_flat = [W[i][c] for i in range(14) for c in range(50)]
    
    return {"U": U_flat, "V": V_flat, "W": W_flat}
