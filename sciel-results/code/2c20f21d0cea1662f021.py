import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # We will use Strassen's algorithm for 2x2x2 matrix multiplication.
    # The problem is 2x4 by 4x6.
    # We can view the 2x4 matrix as two 2x2 blocks side-by-side: A = [A1 A2]
    # We can view the 4x6 matrix as two 2x3 blocks stacked vertically: B = [B1; B2]
    # The product C = A * B = A1 * B1 + A2 * B2.
    # Each term A1*B1 and A2*B2 is a 2x2 by 2x3 multiplication.
    # To use Strassen (which is for 2x2 * 2x2), we can pad the 2x3 matrices with a zero column to make them 2x2.
    # Let B1_pad = [B1, 0] (2x2). Then A1 * B1_pad = [A1*B1, 0]. The first column of this result is A1*B1.
    # Similarly for A2 * B2.
    # So we perform two independent 2x2 * 2x2 Strassen multiplications.
    # Each Strassen multiplication has rank 7.
    # Total rank = 7 + 7 = 14.
    
    # Let's verify the rank minimality.
    # Lower bound for 2x4*4x6 is not well established to be > 14, but 14 is significantly better than 48.
    # Is it possible to do better?
    # If we used 4x4 Strassen? No, 2x4 is not square.
    # What if we treat it as 2x2 * 2x2 blocks differently?
    # The structure is fixed: A is 2x4, B is 4x6.
    # Splitting A into 2x2 blocks horizontally and B into 2x3 blocks vertically is natural.
    # Padding 2x3 to 2x2 is a standard trick.
    # The resulting tensor rank is sum of ranks of the parts.
    # Rank of 2x2*2x2 is 7 (Strassen).
    # So total rank 14.
    
    # Let's construct the matrices U, V, W for rank 14.
    r = 14
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Indices:
    # A is 2x4. i in 0..1, j in 0..3. Index in U: i*m + j = i*4 + j.
    # B is 4x6. j in 0..3, k in 0..5. Index in V: j*p + k = j*6 + k.
    # C is 2x6. i in 0..1, k in 0..5. Index in W: k*n + i = k*2 + i.
    
    # Part 1: A1 (rows 0, cols 0,1) * B1 (rows 0,1; cols 0,1)
    # A1 indices in U: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B1 indices in V: (0,0)->0, (0,1)->6, (1,0)->6, (1,1)->12 
    # Wait, B1 is top-left 2x2 of B (4x6).
    # B rows 0,1. Cols 0,1.
    # V index for B[row, col] is row*6 + col.
    # B[0,0]->0, B[0,1]->1, B[1,0]->6, B[1,1]->7.
    # C1 is top-left 2x2 of C (2x6).
    # C rows 0,1. Cols 0,1.
    # W index for C[row, col] is col*2 + row.
    # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3.
    
    # Part 2: A2 (rows 0, cols 2,3) * B2 (rows 2,3; cols 0,1)
    # A2 indices in U: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # B2 is bottom-left 2x2 of B.
    # B rows 2,3. Cols 0,1.
    # B[2,0]->12, B[2,1]->13, B[3,0]->18, B[3,1]->19.
    # C2 is top-left 2x2 of C (same C, but product of A2*B2).
    # C indices are same as C1.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these 7 multiplications for Part 1 and Part 2 into the global U, V, W.
    # For each part, we allocate 7 columns in U, V, W.
    # Part 1 uses columns 0..6.
    # Part 2 uses columns 7..13.
    
    def add_strassen_block(offset, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Adds Strassen factors for a 2x2 * 2x2 block.
        A is indexed by A_rows, A_cols in the global A matrix.
        B is indexed by B_rows, B_cols in the global B matrix.
        C is indexed by C_rows, C_cols in the global C matrix.
        offset: column offset in U, V, W.
        """
        # Local indices 0,1 for rows and cols of the 2x2 blocks
        # A11 = A[A_rows[0], A_cols[0]]
        # A12 = A[A_rows[0], A_cols[1]]
        # A21 = A[A_rows[1], A_cols[0]]
        # A22 = A[A_rows[1], A_cols[1]]
        
        # B11 = B[B_rows[0], B_cols[0]]
        # B12 = B[B_rows[0], B_cols[1]]
        # B21 = B[B_rows[1], B_cols[0]]
        # B22 = B[B_rows[1], B_cols[1]]
        
        # C11 = C[C_rows[0], C_cols[0]]
        # C12 = C[C_rows[0], C_cols[1]]
        # C21 = C[C_rows[1], C_cols[0]]
        # C22 = C[C_rows[1], C_cols[1]]
        
        # Helper to get global indices
        def idx_U(r, c): return r * m + c
        def idx_V(r, c): return r * p + c
        def idx_W(r, c): return c * n + r
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22
        U[idx_U(A_rows[0], A_cols[0]), offset + 0] += 1
        U[idx_U(A_rows[1], A_cols[1]), offset + 0] += 1
        # V: B11, B22
        V[idx_V(B_rows[0], B_cols[0]), offset + 0] += 1
        V[idx_V(B_rows[1], B_cols[1]), offset + 0] += 1
        # W: C11, C22
        W[idx_W(C_rows[0], C_cols[0]), offset + 0] += 1
        W[idx_W(C_rows[1], C_cols[1]), offset + 0] += 1
        
        # M2 = (A21+A22)B11
        U[idx_U(A_rows[1], A_cols[0]), offset + 1] += 1
        U[idx_U(A_rows[1], A_cols[1]), offset + 1] += 1
        V[idx_V(B_rows[0], B_cols[0]), offset + 1] += 1
        W[idx_W(C_rows[1], C_cols[0]), offset + 1] += 1 # C21
        W[idx_W(C_rows[1], C_cols[1]), offset + 1] -= 1 # C22 (part of -M2)
        
        # M3 = A11(B21-B22)
        U[idx_U(A_rows[0], A_cols[0]), offset + 2] += 1
        V[idx_V(B_rows[1], B_cols[0]), offset + 2] += 1
        V[idx_V(B_rows[1], B_cols[1]), offset + 2] -= 1
        W[idx_W(C_rows[0], C_cols[1]), offset + 2] += 1 # C12
        W[idx_W(C_rows[1], C_cols[1]), offset + 2] += 1 # C22 (part of +M3)
        
        # M4 = A22(B11-B21)
        U[idx_U(A_rows[1], A_cols[1]), offset + 3] += 1
        V[idx_V(B_rows[0], B_cols[0]), offset + 3] += 1
        V[idx_V(B_rows[1], B_cols[0]), offset + 3] -= 1
        W[idx_W(C_rows[0], C_cols[0]), offset + 3] += 1 # C11 (part of +M4)
        W[idx_W(C_rows[1], C_cols[0]), offset + 3] += 1 # C21 (part of +M4)
        
        # M5 = (A11+A12)B22
        U[idx_U(A_rows[0], A_cols[0]), offset + 4] += 1
        U[idx_U(A_rows[0], A_cols[1]), offset + 4] += 1
        V[idx_V(B_rows[1], B_cols[1]), offset + 4] += 1
        W[idx_W(C_rows[0], C_cols[0]), offset + 4] -= 1 # C11 (part of -M5)
        W[idx_W(C_rows[0], C_cols[1]), offset + 4] += 1 # C12 (part of +M5)
        
        # M6 = (A21-A11)(B11+B12)
        U[idx_U(A_rows[1], A_cols[0]), offset + 5] += 1
        U[idx_U(A_rows[0], A_cols[0]), offset + 5] -= 1
        V[idx_V(B_rows[0], B_cols[0]), offset + 5] += 1
        V[idx_V(B_rows[0], B_cols[1]), offset + 5] += 1
        W[idx_W(C_rows[1], C_cols[1]), offset + 5] += 1 # C22 (part of +M6)
        
        # M7 = (A12-A22)(B21+B22)
        U[idx_U(A_rows[0], A_cols[1]), offset + 6] += 1
        U[idx_U(A_rows[1], A_cols[1]), offset + 6] -= 1
        V[idx_V(B_rows[1], B_cols[0]), offset + 6] += 1
        V[idx_V(B_rows[1], B_cols[1]), offset + 6] += 1
        W[idx_W(C_rows[0], C_cols[0]), offset + 6] += 1 # C11 (part of +M7)
        
    # Part 1: A1 (rows 0, cols 0,1) * B1 (rows 0,1; cols 0,1) -> C1 (rows 0,1; cols 0,1)
    # Note: C is 2x6. We only care about cols 0,1 for the first part?
    # A1*B1 contributes to C[:, 0:2] (cols 0 and 1).
    # A2*B2 contributes to C[:, 0:2] (cols 0 and 1).
    # What about C[:, 2:6]?
    # A1*B1 is 2x2. It only affects cols 0,1 of C.
    # A2*B2 is 2x2. It only affects cols 0,1 of C.
    # Wait, B1 is 2x3 (rows 0,1, cols 0,1,2).
    # If we pad B1 to 2x2, we lose col 2.
    # We need to handle the full 4x6 matrix.
    
    # Let's re-evaluate the blocking.
    # A is 2x4. B is 4x6.
    # We can split B into 3 blocks of 2 columns: B = [B1 B2 B3] where Bi is 4x2.
    # Then C = A * B1 + A * B2 + A * B3.
    # Each A * Bi is 2x4 * 4x2 = 2x2.
    # We can use Strassen for each 2x4 * 4x2?
    # 2x4 * 4x2 can be seen as (2x2 * 2x2) + (2x2 * 2x2) if we split A horizontally?
    # A = [A1 A2] (2x2 each).
    # Bi = [Bi_top; Bi_bottom] (2x2 each).
    # A * Bi = A1 * Bi_top + A2 * Bi_bottom.
    # Each term is 2x2 * 2x2.
    # So for each Bi, we need 2 Strassen multiplications.
    # Total 3 * 2 = 6 Strassen multiplications.
    # Rank = 6 * 7 = 42.
    # This is the same as the 2x2 Strassen result mentioned in the knowledge (42).
    # The current program uses schoolbook (48).
    # The knowledge says Strassen (42) is an improvement.
    # Can we do better than 42?
    # The directive suggests 3x3 or 3x4 blocks.
    # But 2x4 and 4x6 are small.
    # Let's stick to the 42 rank solution first, as it's a known improvement over 48.
    # Then we can try to optimize further if time permits, but 42 is a solid structural change.
    
    # Let's implement the 42 rank solution.
    # 6 Strassen blocks.
    # Block (i, j) where i in 0..1 (row of A block), j in 0..2 (col block of B).
    # A block i: rows 0, cols 2*i, 2*i+1.
    # B block j: rows 2*i, 2*i+1; cols 2*j, 2*j+1.
    # C block: rows 0, cols 2*j, 2*j+1.
    
    # Wait, the indices in add_strassen_block above assume 2x2 blocks.
    # Let's verify the indices for the 6 blocks.
    
    # Block 0: i=0, j=0
    # A: rows 0, cols 0,1
    # B: rows 0,1; cols 0,1
    # C: rows 0, cols 0,1
    add_strassen_block(0, [0], [0, 1], [0, 1], [0, 1], [0], [0, 1])
    
    # Block 1: i=0, j=1
    # A: rows 0, cols 0,1
    # B: rows 0,1; cols 2,3
    # C: rows 0, cols 2,3
    add_strassen_block(7, [0], [0, 1], [0, 1], [2, 3], [0], [2, 3])
    
    # Block 2: i=0, j=2
    # A: rows 0, cols 0,1
    # B: rows 0,1; cols 4,5
    # C: rows 0, cols 4,5
    add_strassen_block(14, [0], [0, 1], [0, 1], [4, 5], [0], [4, 5])
    
    # Block 3: i=1, j=0
    # A: rows 0, cols 2,3
    # B: rows 2,3; cols 0,1
    # C: rows 0, cols 0,1
    add_strassen_block(21, [0], [2, 3], [2, 3], [0, 1], [0], [0, 1])
    
    # Block 4: i=1, j=1
    # A: rows 0, cols 2,3
    # B: rows 2,3; cols 2,3
    # C: rows 0, cols 2,3
    add_strassen_block(28, [0], [2, 3], [2, 3], [2, 3], [0], [2, 3])
    
    # Block 5: i=1, j=2
    # A: rows 0, cols 2,3
    # B: rows 2,3; cols 4,5
    # C: rows 0, cols 4,5
    add_strassen_block(35, [0], [2, 3], [2, 3], [4, 5], [0], [4, 5])
    
    # Total rank = 6 * 7 = 42.
    # U shape (8, 42), V shape (24, 42), W shape (12, 42).
    
    # Check time
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
