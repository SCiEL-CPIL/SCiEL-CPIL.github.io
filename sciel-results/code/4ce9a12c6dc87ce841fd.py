"""Program for matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block (1,2): A1 (1x4) * B1 (4x2) -> 1x2, rank 8
    # A1 = A[0,0:4], B1 = B[0:4, 0:2]
    for j in range(4):
        for k in range(2):
            U[0 * 4 + j][col] = 1.0  # A1 row 0, col j
            V[j * 5 + k][col] = 1.0  # B1 row j, col k
            W[k * 3 + 0][col] = 1.0  # C col k, row 0
            col += 1
    
    # Block (1,2): A1 (1x4) * B2 (4x2) -> 1x2, rank 8
    # B2 = B[0:4, 2:4]
    for j in range(4):
        for k in range(2):
            U[0 * 4 + j][col] = 1.0
            V[j * 5 + (2 + k)][col] = 1.0
            W[(2 + k) * 3 + 0][col] = 1.0
            col += 1
    
    # Block (1,1): A1 (1x4) * B3 (4x1) -> 1x1, rank 4
    # B3 = B[0:4, 4:5]
    for j in range(4):
        U[0 * 4 + j][col] = 1.0
        V[j * 5 + 4][col] = 1.0
        W[4 * 3 + 0][col] = 1.0
        col += 1
    
    # Block (2,2): A2 (2x4) * B1 (4x2) -> 2x2, rank 14
    # A2 = A[1:3, 0:4], split into A2a (2x2) and A2b (2x2)
    # B1 = B[0:4, 0:2], split into B1a (2x2) and B1b (2x2)
    # Product = A2a*B1a + A2b*B1b
    # Each M_{2,2,2} uses Strassen rank 7
    
    # Strassen for M_{2,2,2}: 7 terms
    # A2a * B1a: A2a = A[1:3, 0:2], B1a = B[0:2, 0:2]
    # Strassen terms:
    # M1 = (A11+A22)*(B12+B21)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 + M6
    
    def strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, base_col):
        """Add Strassen decomposition for a 2x2 * 2x2 block."""
        nonlocal col
        # A indices: rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1
        # B indices: rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1
        # C indices: rows c_row_start..c_row_start+1, cols c_col_start..c_col_start+1
        
        # M1 = (A11+A22)*(B12+B21)
        U[a_row_start * 4 + a_col_start][base_col] = 1.0
        U[(a_row_start+1) * 4 + (a_col_start+1)][base_col] = 1.0
        V[(b_row_start+1) * 5 + b_col_start][base_col] = 1.0
        V[b_row_start * 5 + (b_col_start+1)][base_col] = 1.0
        W[(b_col_start+1) * 3 + (c_row_start+1)][base_col] = 1.0
        W[(b_col_start) * 3 + c_row_start][base_col] = 1.0
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(a_row_start+1) * 4 + a_col_start][base_col] = 1.0
        U[(a_row_start+1) * 4 + (a_col_start+1)][base_col] = 1.0
        V[b_row_start * 5 + b_col_start][base_col] = 1.0
        W[b_col_start * 3 + (c_row_start+1)][base_col] = 1.0
        col += 1
        
        # M3 = A11*(B11-B21)
        U[a_row_start * 4 + a_col_start][base_col] = 1.0
        V[b_row_start * 5 + b_col_start][base_col] = 1.0
        V[(b_row_start+1) * 5 + b_col_start][base_col] = -1.0
        W[b_col_start * 3 + c_row_start][base_col] = 1.0
        col += 1
        
        # M4 = A22*(B22-B12)
        U[(a_row_start+1) * 4 + (a_col_start+1)][base_col] = 1.0
        V[(b_row_start+1) * 5 + (b_col_start+1)][base_col] = 1.0
        V[b_row_start * 5 + (b_col_start+1)][base_col] = -1.0
        W[(b_col_start+1) * 3 + (c_row_start+1)][base_col] = 1.0
        col += 1
        
        # M5 = (A11+A12)*B22
        U[a_row_start * 4 + a_col_start][base_col] = 1.0
        U[a_row_start * 4 + (a_col_start+1)][base_col] = 1.0
        V[(b_row_start+1) * 5 + (b_col_start+1)][base_col] = 1.0
        W[(b_col_start+1) * 3 + (c_row_start+1)][base_col] = 1.0
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(a_row_start+1) * 4 + a_col_start][base_col] = 1.0
        U[a_row_start * 4 + a_col_start][base_col] = -1.0
        V[b_row_start * 5 + b_col_start][base_col] = 1.0
        V[b_row_start * 5 + (b_col_start+1)][base_col] = 1.0
        W[b_col_start * 3 + (c_row_start+1)][base_col] = 1.0
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[a_row_start * 4 + (a_col_start+1)][base_col] = 1.0
        U[(a_row_start+1) * 4 + (a_col_start+1)][base_col] = -1.0
        V[(b_row_start+1) * 5 + b_col_start][base_col] = 1.0
        V[(b_row_start+1) * 5 + (b_col_start+1)][base_col] = 1.0
        W[(b_col_start+1) * 3 + (c_row_start+1)][base_col] = 1.0
        col += 1
        
        # Now we need to combine these 7 terms to get the 4 C entries
        # But wait - in this formulation, each "multiplication" is one rank-1 term.
        # The C entries are linear combinations of the M terms.
        # In the tensor decomposition, we need to express C[i,j] = sum_s U[i,j;s]*V[...;s]*W[...;s]
        # The W matrix handles the linear combination for C.
        # Let me reconsider the Strassen formulation.
        
        return base_col
    
    # I need to redo this. The Strassen algorithm computes 7 products M1..M7,
    # and then C is a linear combination of these. In the tensor decomposition,
    # each Mi corresponds to a set of rank-1 terms, and the W matrix encodes
    # the linear combination.
    
    # Let me restart the Strassen block more carefully.
    
    # For A (2x2) * B (2x2) = C (2x2):
    # M1 = (A00+A11)*(B01+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B00-B10)
    # M4 = A11*(B11-B01)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M2 + M6
    
    # Each Mi is a product of two linear combinations. In the tensor:
    # Mi = (sum of A terms) * (sum of B terms)
    # This expands to multiple rank-1 terms.
    
    # Actually, in the standard tensor decomposition of matrix multiplication,
    # each "multiplication" in the algorithm corresponds to ONE rank-1 term
    # where U, V, W are the coefficient vectors for that multiplication.
    
    # For M1 = (A00+A11)*(B01+B10):
    # U has 1 at (0,0) and (1,1)
    # V has 1 at (0,1) and (1,0)
    # W has 1 at (0,0) [for C00], 1 at (0,1) [for C01]... 
    # Wait, W should encode which C entries this contributes to.
    
    # C00 gets +1*M1, C01 gets 0*M1, C10 gets 0*M1, C11 gets +1*M1
    # So W for M1: W[0*2+0] = 1, W[1*2+1] = 1 (in local 2x2 coords)
    
    # Hmm, but in our global indexing, W has shape (p*n, rank) = (5*3, rank) = (15, rank)
    # W[k*n + i] where k is column of C, i is row of C.
    
    # Let me define the Strassen block properly.
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        """Add 7 rank-1 terms for Strassen 2x2*2x2.
        a_r0, a_c0: top-left of A block (2x2)
        b_r0, b_c0: top-left of B block (2x2)
        c_r0, c_c0: top-left of C block (2x2)
        """
        nonlocal col
        # M1 = (A00+A11)*(B01+B10) -> contributes to C00 (+1), C11 (+1)
        U[a_r0 * 4 + a_c0][col] = 1.0
        U[(a_r0+1) * 4 + (a_c0+1)][col] = 1.0
        V[b_r0 * 5 + (b_c0+1)][col] = 1.0
        V[(b_r0+1) * 5 + b_c0][col] = 1.0
        W[(c_c0) * 3 + c_r0][col] = 1.0
        W[(c_c0+1) * 3 + (c_r0+1)][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)*B00 -> contributes to C10 (+1), C11 (+1)
        U[(a_r0+1) * 4 + a_c0][col] = 1.0
        U[(a_r0+1) * 4 + (a_c0+1)][col] = 1.0
        V[b_r0 * 5 + b_c0][col] = 1.0
        W[(c_c0) * 3 + (c_r0+1)][col] = 1.0
        W[(c_c0+1) * 3 + (c_r0+1)][col] = 1.0
        col += 1
        
        # M3 = A00*(B00-B10) -> contributes to C01 (+1)
        U[a_r0 * 4 + a_c0][col] = 1.0
        V[b_r0 * 5 + b_c0][col] = 1.0
        V[(b_r0+1) * 5 + b_c0][col] = -1.0
        W[(c_c0+1) * 3 + c_r0][col] = 1.0
        col += 1
        
        # M4 = A11*(B11-B01) -> contributes to C00 (-1), C01 (+1)
        U[(a_r0+1) * 4 + (a_c0+1)][col] = 1.0
        V[(b_r0+1) * 5 + (b_c0+1)][col] = 1.0
        V[b_r0 * 5 + (b_c0+1)][col] = -1.0
        W[(c_c0) * 3 + c_r0][col] = -1.0
        W[(c_c0+1) * 3 + c_r0][col] = 1.0
        col += 1
        
        # M5 = (A00+A01)*B11 -> contributes to C00 (+1), C10 (+1)
        U[a_r0 * 4 + a_c0][col] = 1.0
        U[a_r0 * 4 + (a_c0+1)][col] = 1.0
        V[(b_r0+1) * 5 + (b_c0+1)][col] = 1.0
        W[(c_c0) * 3 + c_r0][col] = 1.0
        W[(c_c0) * 3 + (c_r0+1)][col] = 1.0
        col += 1
        
        # M6 = (A10-A00)*(B00+B01) -> contributes to C11 (+1)
        U[(a_r0+1) * 4 + a_c0][col] = 1.0
        U[a_r0 * 4 + a_c0][col] = -1.0
        V[b_r0 * 5 + b_c0][col] = 1.0
        V[b_r0 * 5 + (b_c0+1)][col] = 1.0
        W[(c_c0+1) * 3 + (c_r0+1)][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)*(B10+B11) -> contributes to C00 (+1)
        U[a_r0 * 4 + (a_c0+1)][col] = 1.0
        U[(a_r0+1) * 4 + (a_c0+1)][col] = -1.0
        V[(b_r0+1) * 5 + b_c0][col] = 1.0
        V[(b_r0+1) * 5 + (b_c0+1)][col] = 1.0
        W[(c_c0) * 3 + c_r0][col] = 1.0
        col += 1
        
        return col
    
    # Now add the Strassen blocks
    # Block A2*B1: A2 = A[1:3, 0:4], B1 = B[0:4, 0:2]
    # Split: A2a = A[1:3, 0:2], A2b = A[1:3, 2:4]
    #        B1a = B[0:2, 0:2], B1b = B[2:4, 0:2]
    # C for A2*B1 is C[1:3, 0:2]
    
    # A2a * B1a -> C[1:3, 0:2]
    add_strassen(1, 0, 0, 0, 1, 0, col)
    
    # A2b * B1b -> C[1:3, 0:2]
    add_strassen(1, 2, 2, 0, 1, 0, col)
    
    # Block A2*B2: A2 = A[1:3, 0:4], B2 = B[0:4, 2:4]
    # A2a = A[1:3, 0:2], A2b = A[1:3, 2:4]
    # B2a = B[0:2, 2:4], B2b = B[2:4, 2:4]
    # C for A2*B2 is C[1:3, 2:4]
    
    add_strassen(1, 0, 0, 2, 1, 2, col)
    add_strassen(1, 2, 2, 2, 1, 2, col)
    
    # Block A2*B3: A2 (2x4) * B3 (4x1) -> 2x1, rank 8
    for j in range(4):
        for i in range(2):
            U[(1 + i) * 4 + j][col] = 1.0
            V[j * 5 + 4][col] = 1.0
            W[4 * 3 + (1 + i)][col] = 1.0
            col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
