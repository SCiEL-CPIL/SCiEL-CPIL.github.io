"""Matrix multiplication 4x4x5 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2(U_rows, V_rows, W_rows, offset):
    """
    Add Strassen 2x2x2 decomposition (7 terms) for a 2x2 block.
    U_rows: list of 2 row indices in U space (each row is 2 elements wide in block)
    V_rows: list of 2 row indices in V space
    W_rows: list of 2 row indices in W space
    offset: column offset in the factor matrices
    Returns: new offset (offset + 7)
    """
    # Strassen 7 multiplications for 2x2*2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    u0, u1 = U_rows
    v0, v1 = V_rows
    w0, w1 = W_rows
    
    # U indices: u0*2+0, u0*2+1, u1*2+0, u1*2+1
    # V indices: v0*2+0, v0*2+1, v1*2+0, v1*2+1
    # W indices: w0*2+0, w0*2+1, w1*2+0, w1*2+1
    
    # M1 = (A11+A22)(B11+B22)
    U[u0*2+0][offset] = 1; U[u1*2+1][offset] = 1
    V[v0*2+0][offset] = 1; V[v1*2+1][offset] = 1
    W[w0*2+0][offset] = 1; W[w1*2+1][offset] = 1
    offset += 1
    
    # M2 = (A21+A22)B11
    U[u1*2+0][offset] = 1; U[u1*2+1][offset] = 1
    V[v0*2+0][offset] = 1
    W[w1*2+0][offset] = 1
    offset += 1
    
    # M3 = A11(B21-B22)
    U[u0*2+0][offset] = 1
    V[v1*2+0][offset] = 1; V[v1*2+1][offset] = -1
    W[w0*2+1][offset] = 1; W[w1*2+0][offset] = 1
    offset += 1
    
    # M4 = A22(B11-B21)
    U[u1*2+1][offset] = 1
    V[v0*2+0][offset] = 1; V[v1*2+0][offset] = -1
    W[w0*2+0][offset] = 1; W[w1*2+0][offset] = 1
    offset += 1
    
    # M5 = (A11+A12)B22
    U[u0*2+0][offset] = 1; U[u0*2+1][offset] = 1
    V[v1*2+1][offset] = 1
    W[w0*2+0][offset] = -1; W[w1*2+1][offset] = 1
    offset += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[u1*2+0][offset] = 1; U[u0*2+0][offset] = -1
    V[v0*2+0][offset] = 1; V[v0*2+1][offset] = 1
    W[w1*2+1][offset] = 1
    offset += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[u0*2+1][offset] = 1; U[u1*2+1][offset] = -1
    V[v1*2+0][offset] = 1; V[v1*2+1][offset] = 1
    W[w0*2+1][offset] = 1; W[w1*2+1][offset] = -1
    offset += 1
    
    return offset

def _strassen_4x4(U, V, W, offset):
    """
    Add Strassen 4x4x4 decomposition (49 terms) using recursion.
    """
    # 4x4 can be decomposed into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # U space: rows 0-3, each row has 4 elements (for 4x4 A)
    # Actually, U has n*m = 16 rows for 4x4, but we're doing block structure
    # Let me reconsider the indexing
    
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # U has 16 rows (i*m+j = i*4+j), V has 16 rows (j*p+k = j*4+k), W has 16 rows (k*n+i = k*4+i)
    
    # Block structure:
    # A11: rows 0-1, cols 0-1 -> U indices: 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> U indices: 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> U indices: 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> U indices: 10,11,14,15
    
    # B11: rows 0-1, cols 0-1 -> V indices: 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> V indices: 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> V indices: 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> V indices: 10,11,14,15
    
    # C11: rows 0-1, cols 0-1 -> W indices: 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> W indices: 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> W indices: 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> W indices: 10,11,14,15
    
    # Use Strassen on the 2x2 block structure, where each "element" is a 2x2 matrix
    # This gives 7 * 7 = 49 terms (each 2x2 Strassen has 7 terms, and we have 7 such operations)
    
    # Actually, the standard Strassen for 4x4x4 uses 49 multiplications.
    # Let me implement it by applying Strassen at the 2x2 block level, where each
    # block operation uses the 2x2x2 Strassen (7 terms).
    
    # The 7 Strassen multiplications at the 2x2 block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each of these is a 2x2 * 2x2 multiplication, which needs 7 scalar multiplications
    # So total = 7 * 7 = 49
    
    # Let's define the U, V, W index sets for each block
    # U block rows (each block is 2x2 = 4 rows in U space):
    # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
    # V block rows:
    # B11: [0,1,4,5], B12: [2,3,6,7], B21: [8,9,12,13], B22: [10,11,14,15]
    # W block rows:
    # C11: [0,1,4,5], C12: [2,3,6,7], C21: [8,9,12,13], C22: [10,11,14,15]
    
    # For each Strassen term, we need to do a 2x2*2x2 multiplication
    # which itself uses 7 scalar terms via the 2x2 Strassen
    
    # This is getting complex. Let me use a different approach:
    # Directly enumerate the 49 terms of the 4x4x4 Strassen decomposition.
    
    # Actually, let me just use the recursive approach properly.
    
    # I'll define helper to add a 2x2x2 Strassen block
    # where the U, V, W "rows" are lists of actual row indices
    
    def add_2x2_strassen(U_rows, V_rows, W_rows, col_offset):
        """Add 7-term Strassen for 2x2*2x2.
        U_rows: 4 indices (2x2 block of U)
        V_rows: 4 indices (2x2 block of V)
        W_rows: 4 indices (2x2 block of W)
        """
        # U_rows[0], U_rows[1] = first 2 rows (row 0 of 2x2 block)
        # U_rows[2], U_rows[3] = next 2 rows (row 1 of 2x2 block)
        # Actually, let me use a cleaner mapping:
        # For a 2x2 block of A (which is part of the 4x4), the 4 U indices are:
        #   top-left, top-right, bottom-left, bottom-right
        # Similarly for V and W.
        
        u00, u01, u10, u11 = U_rows  # A block
        v00, v01, v10, v11 = V_rows  # B block
        w00, w01, w10, w11 = W_rows  # C block
        
        # 7 Strassen terms:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[u00][col_offset] = 1; U[u11][col_offset] = 1
        V[v00][col_offset] = 1; V[v11][col_offset] = 1
        W[w00][col_offset] = 1; W[w11][col_offset] = 1
        col_offset += 1
        
        # M2 = (A10+A11)B00 -> C10 += M2
        U[u10][col_offset] = 1; U[u11][col_offset] = 1
        V[v00][col_offset] = 1
        W[w10][col_offset] = 1
        col_offset += 1
        
        # M3 = A00(B10-B11) -> C01 += M3, C10 += M3
        U[u00][col_offset] = 1
        V[v10][col_offset] = 1; V[v11][col_offset] = -1
        W[w01][col_offset] = 1; W[w10][col_offset] = 1
        col_offset += 1
        
        # M4 = A11(B00-B10) -> C00 += M4, C10 += M4
        U[u11][col_offset] = 1
        V[v00][col_offset] = 1; V[v10][col_offset] = -1
        W[w00][col_offset] = 1; W[w10][col_offset] = 1
        col_offset += 1
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C11 += M5
        U[u00][col_offset] = 1; U[u01][col_offset] = 1
        V[v11][col_offset] = 1
        W[w00][col_offset] = -1; W[w11][col_offset] = 1
        col_offset += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        U[u10][col_offset] = 1; U[u00][col_offset] = -1
        V[v00][col_offset] = 1; V[v01][col_offset] = 1
        W[w11][col_offset] = 1
        col_offset += 1
        
        # M7 = (A01-A11)(B10+B11) -> C01 += M7, C11 -= M7
        U[u01][col_offset] = 1; U[u11][col_offset] = -1
        V[v10][col_offset] = 1; V[v11][col_offset] = 1
        W[w01][col_offset] = 1; W[w11][col_offset] = -1
        col_offset += 1
        
        return col_offset
    
    # Now the 4x4x4 Strassen:
    # A blocks: A11=[0,2,4,6], A12=[1,3,5,7], A21=[8,10,12,14], A22=[9,11,13,15]
    # Wait, let me get the indexing right.
    # A is 4x4, U row index = i*m + j = i*4 + j where i is row, j is col
    # A11 (rows 0-1, cols 0-1): U indices 0,1,4,5
    # A12 (rows 0-1, cols 2-3): U indices 2,3,6,7
    # A21 (rows 2-3, cols 0-1): U indices 8,9,12,13
    # A22 (rows 2-3, cols 2-3): U indices 10,11,14,15
    
    # For the 2x2 block structure, I need to map each 2x2 block to 4 U indices
    # arranged as [top-left, top-right, bottom-left, bottom-right]
    
    # A11: rows 0,1; cols 0,1 -> U[0]=A[0,0], U[1]=A[0,1], U[4]=A[1,0], U[5]=A[1,1]
    # So A11 block = [0, 1, 4, 5] (top-left, top-right, bottom-left, bottom-right)
    
    # A12: rows 0,1; cols 2,3 -> U[2]=A[0,2], U[3]=A[0,3], U[6]=A[1,2], U[7]=A[1,3]
    # A12 block = [2, 3, 6, 7]
    
    # A21: rows 2,3; cols 0,1 -> U[8]=A[2,0], U[9]=A[2,1], U[12]=A[3,0], U[13]=A[3,1]
    # A21 block = [8, 9, 12, 13]
    
    # A22: rows 2,3; cols 2,3 -> U[10]=A[2,2], U[11]=A[2,3], U[14]=A[3,2], U[15]=A[3,3]
    # A22 block = [10, 11, 14, 15]
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    # B is 4x4, V row index = j*p + k = j*4 + k where j is row, k is col
    # B11 (rows 0-1, cols 0-1): V indices 0,1,4,5
    # B12 (rows 0-1, cols 2-3): V indices 2,3,6,7
    # B21 (rows 2-3, cols 0-1): V indices 8,9,12,13
    # B22 (rows 2-3, cols 2-3): V indices 10,11,14,15
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    # C is 4x4, W row index = k*n + i = k*4 + i where k is col, i is row
    # C11 (rows 0-1, cols 0-1): W indices 0,1,4,5
    # C12 (rows 0-1, cols 2-3): W indices 2,3,6,7
    # C21 (rows 2-3, cols 0-1): W indices 8,9,12,13
    # C22 (rows 2-3, cols 2-3): W indices 10,11,14,15
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # Now apply the 7 Strassen terms, each being a 2x2*2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # This means we need to compute (A11+A22) * (B11+B22)
    # The U part: A11+A22 means for each position, sum the two blocks
    # But in our tensor decomposition, we can't directly add blocks in U/V.
    # Instead, we need to expand this into individual scalar multiplications.
    
    # The product (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each of these is a 2x2*2x2 multiplication, which can be done with 7 terms each
    # But that would be 4*7=28 terms just for M1, which is too many.
    
    # The correct approach: the Strassen algorithm at the 2x2 block level
    # treats each block as an "element" and uses 7 block multiplications.
    # Each block multiplication (2x2 * 2x2) requires 7 scalar multiplications.
    # So total = 7 * 7 = 49.
    
    # But we need to be careful: the block-level Strassen combines blocks first,
    # then multiplies. So M1 = (A11+A22)(B11+B22) is ONE block multiplication
    # where the "elements" are 2x2 matrices.
    
    # To implement this, for M1, we compute (A11+A22) * (B11+B22) using
    # the 2x2 Strassen, where the "entries" of the 2x2 matrices are the
    # 2x2 blocks themselves.
    
    # This means:
    # M1_00 = ((A11+A22)_00 + (A11+A22)_11) * ((B11+B22)_00 + (B11+B22)_11)
    #         = (A11_00 + A22_00 + A11_11 + A22_11) * (B11_00 + B22_00 + B11_11 + B22_11)
    
    # This is getting very complex. Let me use a simpler approach:
    # Just use the known 49-term decomposition directly.
    
    # Actually, let me step back and use a completely different strategy:
    # Use the schoolbook for the 4x4x4 part (64 terms) plus 16 for the 5th column = 80.
    # That's not better than the baseline.
    
    # Let me try: use Strassen for 4x4x4 (49 terms) + 16 scalar for col 5 = 65 total.
    
    # I'll implement the 4x4x4 Strassen by directly enumerating the 49 terms.
    # Reference: the 4x4x4 matrix multiplication can be done with 49 multiplications.
    
    # Let me just hardcode the 49-term Strassen decomposition for 4x4x4.
    
    # After research, the standard recursive Strassen for 4x4x4 gives 49 terms.
    # Let me implement it by calling the 2x2 Strassen 7 times, but at the block level.
    
    # I'll restructure: define a function that takes 4 U indices, 4 V indices, 4 W indices
    # representing a 2x2 block multiplication, and adds 7 terms.
    
    # For the 4x4 case, the 7 block-level Strassen terms are:
    # T1 = (A11+A22) * (B11+B22)
    # T2 = (A21+A22) * B11
    # T3 = A11 * (B21-B22)
    # T4 = A22 * (B11-B21)
    # T5 = (A11+A12) * B22
    # T6 = (A21-A11) * (B11+B12)
    # T7 = (A12-A22) * (B21+B22)
    
    # Each Ti is a 2x2 * 2x2 block multiplication.
    # To compute Ti, we use the 2x2 Strassen (7 scalar terms).
    
    # For T1 = (A11+A22) * (B11+B22):
    # Let X = A11+A22 (a 2x2 matrix where X[i,j] = A11[i,j] + A22[i,j])
    # Let Y = B11+B22 (a 2x2 matrix where Y[i,j] = B11[i,j] + B22[i,j])
    # Then T1 = X * Y, computed with 7 scalar multiplications.
    
    # The U indices for X: for each of the 4 positions in the 2x2 block,
    # we have two U indices (one from A11, one from A22) that both contribute.
    # So U for X = A11_indices + A22_indices (8 indices total, but paired)
    
    # This is getting too complex to implement correctly from scratch in this context.
    
    # Let me use a pragmatic approach: implement the 49-term decomposition
    # by directly listing all (U_idx, V_idx, W_idx) triples with coefficients.
    
    # I'll generate them programmatically using the recursive Strassen structure.
    
    # Define the 2x2 Strassen terms as (u_offset, v_offset, w_offset, coeff)
    # where offsets are into the 4-element blocks
    
    strassen_2x2_terms = [
        # (u0, v0, w0, c) for each of the 7 terms
        # M1 = (A00+A11)(B00+B11)
        [(0, 0, 0, 1), (3, 3, 0, 1), (0, 3, 3, 1), (3, 0, 3, 1)],  # expanded? No...
    ]
    
    # This approach is error-prone. Let me use a cleaner recursive implementation.
    
    # I'll represent the decomposition as a list of (U_row, V_row, W_row, coeff) tuples.
    
    def strassen_2x2_decomp():
        """Return list of (u, v, w, c) for 2x2*2x2 Strassen.
        u,v,w are indices 0-3 into the 2x2 blocks (row-major: 00,01,10,11).
        """
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # This expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
        # But in Strassen, we compute it as ONE multiplication of sums.
        # In the tensor decomposition, each "multiplication" is one rank-1 term.
        # So M1 contributes 4 rank-1 terms? No, that's schoolbook.
        
        # The Strassen trick: M1 = (A00+A11)(B00+B11) is ONE scalar multiplication
        # in the algorithm, but in the tensor decomposition, it corresponds to
        # the product of the sum (A00+A11) and the sum (B00+B11).
        # This is NOT a single rank-1 term; it's the sum of 4 rank-1 terms.
        
        # I'm confusing the algorithmic complexity with the tensor rank.
        # The tensor rank of 2x2*2x2 is 7, not 4.
        
        # The 7 rank-1 terms for 2x2*2x2 Strassen:
        # Each term is u_i * v_j * w_k where u_i, v_j, w_k are vectors.
        
        # Let me just use the known 7-term decomposition:
        # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  (A00+A11)*(B00+B11) -> C00,C11
        # But this isn't quite right for the tensor form.
        
        # In the tensor form T[i,j,k] = C[i,j] for A[i,:] * B[:,j]
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11) -> contributes to C00 and C11
        #    U vector: [1,0,0,1], V vector: [1,0,0,1], W vector: [1,0,0,1]
        #    But W should be [1,0,0,1] for C00 and C11? 
        #    C00 is at W index 0, C11 is at W index 3.
        #    So W = [1,0,0,1].
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2 = (A10+A11)*B00 -> C10
            ([1,0,0,0], [0,1,0,-1], [0,1,1,0]),  # M3 = A00*(B10-B11) -> C01,C10
            ([0,0,0,1], [1,0,-1,0], [1,0,1,0]),  # M4 = A11*(B00-B10) -> C00,C10
            ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),  # M5 = (A00+A01)*B11 -> C00-,C11+
            ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),  # M6 = (A10-A00)*(B00+B01) -> C11
            ([0,1,0,-1], [0,0,1,1], [0,1,0,-1]),  # M7 = (A01-A11)*(B10+B11) -> C01,C11-
        ]
        return terms
    
    # Now for 4x4x4, I apply the 2x2 Strassen at the block level.
    # The 7 block-level terms:
    # T1 = (A11+A22)*(B11+B22)
    # T2 = (A21+A22)*B11
    # T3 = A11*(B21-B22)
    # T4 = A22*(B11-B21)
    # T5 = (A11+A12)*B22
    # T6 = (A21-A11)*(B11+B12)
    # T7 = (A12-A22)*(B21+B22)
    
    # Each Ti is a 2x2*2x2 block multiplication, which expands to 7 scalar rank-1 terms.
    # Total: 7*7 = 49 terms.
    
    # To implement: for each block-level term, I need to:
    # 1. Form the U vector (16-dim) for the left block
    # 2. Form the V vector (16-dim) for the right block
    # 3. Form the W vector (16-dim) for the result block
    # 4. Apply the 2x2 Strassen decomposition to get 7 rank-1 terms
    
    # For T1 = (A11+A22)*(B11+B22):
    # U block = A11 + A22 (2x2 matrix)
    # V block = B11 + B22 (2x2 matrix)
    # W block = C11 + C22 (2x2 matrix, since T1 contributes to both C11 and C22)
    
    # The U vector for the full 4x4 A:
    # A11 occupies U indices [0,1,4,5], A22 occupies [10,11,14,15]
    # So U for (A11+A22) has 1s at these 8 indices.
    
    # Similarly for V and W.
    
    # Then apply the 2x2 Strassen terms, where each "2x2 index" maps to
    # a pair of full indices.
    
    # This is very involved. Let me just use the schoolbook for now (rank 80)
    # and accept the baseline score, OR find a simpler improvement.
    
    # Actually, let me just return the schoolbook (rank 80) to be safe,
    # since I can't easily verify the Strassen implementation.
    
    pass

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    # Use schoolbook: rank = n*m*p = 80
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
# EVOLVE-BLOCK-END
