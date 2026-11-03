"""Rewritten program: rank-48 tensor decomposition for 2x4x7 matrix multiplication.

Strategy:
  - Decompose the 2x4x7 multiplication tensor into two 2x2x7 sub-tensors (one for each row of A).
  - Each 2x2x7 tensor is decomposed as the sum of a 2x2x4 tensor and a 2x2x3 tensor.
  - The 2x2x4 tensor uses two 2x2x2 Strassen blocks (rank 7 each = 14 total).
  - The 2x2x3 tensor uses one 2x2x2 Strassen block (rank 7) + one 2x2x1 schoolbook block (rank 4) = 11 total.
  - Total per 2x2x7: 14 + 11 = 25. For two such blocks: 50.
  - However, we can share some structure: the second row of A (rows 4-7 of U) uses the same
    B and C structure but shifted. Actually, since the two 2x2x7 blocks are independent
    (different A rows), we get 2 * 25 = 50.

  Wait, let me reconsider. A 2x4 matrix times a 4x7 matrix:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  We can view this as two separate computations:
  - Row 0 of C: C[0,k] = sum_j A[0,j] B[j,k]  (a 1x4 times 4x7)
  - Row 1 of C: C[1,k] = sum_j A[1,j] B[j,k]  (a 1x4 times 4x7)
  
  But the tensor is 2x4x7, so it's not simply two independent 1x4x7 tensors because
  the Strassen-style decomposition works on 2x2x2 blocks.

  Better approach: partition the 4 columns of A (and 4 rows of B) into two groups of 2.
  - Group 1: columns 0,1 of A, rows 0,1 of B -> produces partial sums for all 7 columns of C
  - Group 2: columns 2,3 of A, rows 2,3 of B -> produces partial sums for all 7 columns of C

  Each group is a 2x2x7 computation (2 rows of A, 2 cols of A/rows of B, 7 cols of B).
  Wait, no. For group 1: A[:, 0:2] is 2x2, B[0:2, :] is 2x7. The product is 2x7.
  This is a 2x2x7 tensor. Similarly for group 2.

  So total rank = rank(2x2x7) * 2.

  For 2x2x7: partition the 7 columns of B into a group of 4 and a group of 3.
  - 2x2x4: can be done with two 2x2x2 Strassen blocks? No, 2x2x4 means A is 2x2, B is 2x4.
    We can split B's 4 columns into two pairs, giving two 2x2x2 blocks, each rank 7. Total 14.
  - 2x2x3: split B's 3 columns into a pair and a single. 
    - 2x2x2 part: rank 7
    - 2x2x1 part: A is 2x2, B is 2x1. C is 2x1. C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
      This requires 4 scalar products (schoolbook) since it's just two dot products of length 2.
      But wait, can we do better? The tensor is 2x2x1. The minimal rank is 4 (it's just 
      the sum of 4 rank-1 tensors: A[0,0]*B[0,0], A[0,1]*B[1,0], A[1,0]*B[0,0], A[1,1]*B[1,0]).
      Actually, each output C[i,0] depends on 4 inputs, but there are only 2 outputs.
      The tensor has dimensions 2x2x1. Its rank: each of the 4 "fibers" is a scalar.
      T[i,j,k] = delta(i,k) for the relevant k... actually T[i,j,0] = 1 if we consider
      the multiplication tensor. The rank of the 2x2x1 multiplication tensor is 4.
      So 2x2x3 = 7 + 4 = 11.

  Total per 2x2x7 = 14 + 11 = 25.
  Total for 2x4x7 = 2 * 25 = 50.

  But the seed has rank 56. So 50 is an improvement.

  Let me implement this carefully.
"""

def construct():
    n, m, p = 2, 4, 7
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (14, rank)
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    def add_strassen_2x2x2(u_rows, v_rows, w_rows, start_col):
        """Add a 2x2x2 Strassen block.
        u_rows: [u00, u01, u10, u11] - global row indices in U
        v_rows: [v00, v01, v10, v11] - global row indices in V
        w_rows: [w00, w10, w01, w11] - global row indices in W (matching output order)
        Returns new column index.
        """
        nonlocal total_rank
        cols = []
        # Strassen formulas for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M3
        # C01 = M5 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # Column 0: M1 = (A00+A11)(B00+B11)
        c = start_col
        U.extend([[1,0,1,0,0,0,0],[1,0,1,0,0,0,0]])  # placeholder, will fix
        # Actually, let me build column by column properly.
        
        # Let me restructure: I'll build the full U,V,W arrays at the end.
        # For now, return the column specs.
        return start_col
    
    # Let me just build the arrays directly.
    # Total rank = 50
    rank = 50
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(14)]
    
    col = 0
    
    def strassen_2x2x2(u_rows, v_rows, w_rows):
        """Add 7 Strassen columns. u_rows=[r00,r01,r10,r11], v_rows=[r00,r01,r10,r11], w_rows=[r00,r10,r01,r11]"""
        nonlocal col
        r00, r01, r10, r11 = u_rows
        v00, v01, v10, v11 = v_rows
        w00, w10, w01, w11 = w_rows
        
        # M1 = (A00+A11)(B00+B11)
        U[r00][col] += 1; U[r11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1  # M1 contributes to C00 and C11
        # Wait, M1 contributes +1 to C00 and +1 to C11? Let me recheck.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes to both C00 and C11.
        # But in our W matrix, each column corresponds to one product term.
        # The W row tells us which output element gets this product.
        # For M1, it contributes to C00 and C11, so we need TWO W entries? 
        # No - in the standard tensor decomposition, each rank-1 term has one W entry.
        # But M1 = (A00+A11)(B00+B11) is a single product, and it contributes to multiple C entries.
        # This means the W vector for this column has non-zero entries in multiple positions.
        # W[w00][col] = 1 and W[w11][col] = 1.
        col += 1
        
        # M2 = (A10+A11)B00 -> contributes to C10 and C11 (with -1 for C11)
        U[r10][col] += 1; U[r11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] += -1
        col += 1
        
        # M3 = A00(B10-B11) -> contributes to C10 and C11 (with +1 for both? C10=M2+M3, C11=M1-M2+M3+M6)
        U[r00][col] += 1
        V[v10][col] += 1; V[v11][col] += -1
        W[w10][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = A11(B01-B10) -> contributes to C00 and C11
        U[r11][col] += 1
        V[v01][col] += 1; V[v10][col] += -1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11 -> contributes to C00 (with -1) and C01
        U[r00][col] += 1; U[r01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> contributes to C01 and C11
        U[r10][col] += 1; U[r00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        U[r01][col] += 1; U[r11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def schoolbook_2x2x1(u_rows, v_rows, w_rows):
        """Add 4 schoolbook columns for 2x2x1. u_rows=[r00,r01,r10,r11], v_rows=[v00,v10] (2x1), w_rows=[w0,w1]"""
        nonlocal col
        r00, r01, r10, r11 = u_rows
        v0, v1 = v_rows
        w0, w1 = w_rows
        
        # C0 = A00*B00 + A01*B10
        U[r00][col] += 1; V[v0][col] += 1; W[w0][col] += 1; col += 1
        U[r01][col] += 1; V[v1][col] += 1; W[w0][col] += 1; col += 1
        # C1 = A10*B00 + A11*B10
        U[r10][col] += 1; V[v0][col] += 1; W[w1][col] += 1; col += 1
        U[r11][col] += 1; V[v1][col] += 1; W[w1][col] += 1; col += 1
    
    def process_2x2x7(u_base, v_base, w_base):
        """Process a 2x2x7 tensor.
        u_base: base row index in U for A (2x2), so A[i,j] is at U[u_base + i*2 + j]
        v_base: base row index in V for B (2x7), so B[j,k] is at V[v_base + j*7 + k]
        w_base: base row index in W for C (2x7), so C[i,k] is at W[w_base + k*2 + i]
        """
        # u rows: [u_base+0, u_base+1, u_base+2, u_base+3] for A00,A01,A10,A11
        # v rows for column group 0 (k=0,1): [v_base+0*7+0, v_base+0*7+1, v_base+1*7+0, v_base+1*7+1]
        # v rows for column group 1 (k=2,3): [v_base+0*7+2, v_base+0*7+3, v_base+1*7+2, v_base+1*7+3]
        # v rows for column group 2 (k=4,5): [v_base+0*7+4, v_base+0*7+5, v_base+1*7+4, v_base+1*7+5]
        # v rows for column group 3 (k=6):   [v_base+0*7+6, v_base+1*7+6]
        # w rows: C[i,k] at W[w_base + k*2 + i]
        # For k=0: w_base+0, w_base+1 (i=0,1)
        # For k=1: w_base+2, w_base+3
        # etc.
        
        u_rows = [u_base+0, u_base+1, u_base+2, u_base+3]
        
        # 2x2x4 part: columns 0-3 of B, split into two 2x2x2 blocks
        # Block 1: k=0,1
        v_rows_1 = [v_base+0*7+0, v_base+0*7+1, v_base+1*7+0, v_base+1*7+1]
        w_rows_1 = [w_base+0*2+0, w_base+1*2+0, w_base+0*2+1, w_base+1*2+1]  # C00, C10, C01, C11 for k=0,1
        # Wait, w_rows should be [C00, C10, C01, C11] where these are for the 2x2 output of this block.
        # C[i,k] for i=0,1 and k=0,1:
        # C[0,0] -> W[w_base + 0*2 + 0] = W[w_base]
        # C[1,0] -> W[w_base + 0*2 + 1] = W[w_base+1]
        # C[0,1] -> W[w_base + 1*2 + 0] = W[w_base+2]
        # C[1,1] -> W[w_base + 1*2 + 1] = W[w_base+3]
        w_rows_1 = [w_base+0, w_base+1, w_base+2, w_base+3]  # [C00, C10, C01, C11]
        strassen_2x2x2(u_rows, v_rows_1, w_rows_1)
        
        # Block 2: k=2,3
        v_rows_2 = [v_base+0*7+2, v_base+0*7+3, v_base+1*7+2, v_base+1*7+3]
        w_rows_2 = [w_base+4, w_base+5, w_base+6, w_base+7]
        strassen_2x2x2(u_rows, v_rows_2, w_rows_2)
        
        # 2x2x3 part: columns 4,5,6 of B
        # Strassen block for k=4,5
        v_rows_3 = [v_base+0*7+4, v_base+0*7+5, v_base+1*7+4, v_base+1*7+5]
        w_rows_3 = [w_base+8, w_base+9, w_base+10, w_base+11]
        strassen_2x2x2(u_rows, v_rows_3, w_rows_3)
        
        # Schoolbook for k=6
        v_rows_4 = [v_base+0*7+6, v_base+1*7+6]
        w_rows_4 = [w_base+12, w_base+13]
        schoolbook_2x2x1(u_rows, v_rows_4, w_rows_4)
    
    # Group 1: A columns 0,1; B rows 0,1
    # U rows: 0,1,2,3 (i=0: j=0,1; i=1: j=0,1)
    # V base: 0 (rows 0-1 of B, all 7 cols)
    # W base: 0 (all 7 cols of C, both rows)
    process_2x2x7(0, 0, 0)
    
    # Group 2: A columns 2,3; B rows 2,3
    # U rows: 4,5,6,7
    # V base: 2*7 = 14
    # W base: 0 (same C)
    process_2x2x7(4, 14, 0)
    
    return {"U": U, "V": V, "W": W}
