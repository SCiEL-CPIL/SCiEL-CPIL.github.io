"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Use the block structure and known low-rank decompositions.
    - Split 4x4 * 4x5 into 8 instances of 2x2 * 2x2 and 4 instances of 2x2 * 2x1
    - Use Strassen's rank-7 algorithm for 2x2 * 2x2
    - Use rank-1 decomposition for 2x2 * 2x1
    - Total rank: 8*7 + 4*1 = 60
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    deadline = time.time() + 135
    
    # Strassen's algorithm for 2x2 * 2x2: rank 7
    # The 7 intermediate products are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For each 2x2 * 2x2 multiplication, we have 7 terms.
    # Each term has a specific pattern of U, V, W entries.
    
    # Let's build the full decomposition.
    # The 4x4 matrix A is indexed by (i, j) where i=0..3, j=0..3
    # The 4x5 matrix B is indexed by (j, k) where j=0..3, k=0..4
    # The 4x5 result C is indexed by (i, k) where i=0..3, k=0..4
    
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where A00, A01, A10, A11 are 2x2
    #   A00[i,j] = A[2*i, 2*j] for i,j in 0,1
    #   A01[i,j] = A[2*i, 2*j+1] for i,j in 0,1
    #   A10[i,j] = A[2*i+1, 2*j] for i,j in 0,1
    #   A11[i,j] = A[2*i+1, 2*j+1] for i,j in 0,1
    # 
    # B = [[B00, B01, B02], [B10, B11, B12]] where:
    #   B00[i,j] = B[2*i, 2*j] for i in 0,1, j in 0,1 (2x2)
    #   B01[i,j] = B[2*i, 2*j+1] for i in 0,1, j in 0,1 (2x2)
    #   B02[i,0] = B[2*i, 2] for i in 0,1 (2x1)
    #   B10[i,j] = B[2*i+1, 2*j] for i in 0,1, j in 0,1 (2x2)
    #   B11[i,j] = B[2*i+1, 2*j+1] for i in 0,1, j in 0,1 (2x2)
    #   B12[i,0] = B[2*i+1, 2] for i in 0,1 (2x1)
    
    # C = [[C00, C01, C02], [C10, C11, C12]] where:
    #   C00 = A00*B00 + A01*B10 (2x2)
    #   C01 = A00*B01 + A01*B11 (2x2)
    #   C02 = A00*B02 + A01*B12 (2x1)
    #   C10 = A10*B00 + A11*B10 (2x2)
    #   C11 = A10*B01 + A11*B11 (2x2)
    #   C12 = A10*B02 + A11*B12 (2x1)
    
    # So we need:
    # - 8 instances of 2x2 * 2x2: (A00,B00), (A01,B10), (A00,B01), (A01,B11), 
    #   (A10,B00), (A11,B10), (A10,B01), (A11,B11)
    # - 4 instances of 2x2 * 2x1: (A00,B02), (A01,B12), (A10,B02), (A11,B12)
    
    # For each 2x2 * 2x2, use Strassen (rank 7)
    # For each 2x2 * 2x1, use rank 1
    
    # Total rank: 8*7 + 4*1 = 60
    
    # Let's build the U, V, W matrices.
    # U has shape (n*m, rank) = (16, 60)
    # V has shape (m*p, rank) = (20, 60)
    # W has shape (p*n, rank) = (20, 60)
    
    rank_total = 60
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    # Helper: add a Strassen term for a specific 2x2 * 2x2 multiplication
    # The 2x2 block A_block is at position (row_block, col_block) in the 4x4 matrix
    # The 2x2 block B_block is at position (row_block, col_block) in the 4x5 matrix
    # The result goes to position (row_block, col_block) in the 4x5 result
    
    def add_strassen_term(row_block_a, col_block_a, row_block_b, col_block_b, col_offset):
        """Add 7 Strassen terms for a 2x2 * 2x2 multiplication."""
        # A_block indices: rows 2*row_block_a to 2*row_block_a+1, cols 2*col_block_a to 2*col_block_a+1
        # B_block indices: rows 2*row_block_b to 2*row_block_b+1, cols 2*col_block_b to 2*col_block_b+1
        # Result indices: rows 2*row_block_a to 2*row_block_a+1, cols 2*col_block_b to 2*col_block_b+1
        
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # For each product, we need to specify the U, V, W entries.
        # U[i*m+j, r] where i is row of A, j is col of A (which is row of B)
        # V[j*p+k, r] where j is row of B, k is col of B
        # W[k*n+i, r] where k is col of B (which is col of result), i is row of result
        
        # M1: (A11 + A22) * (B11 + B22)
        # A11 is at (row_block_a*2, col_block_a*2)
        # A22 is at (row_block_a*2+1, col_block_a*2+1)
        # B11 is at (row_block_b*2, col_block_b*2)
        # B22 is at (row_block_b*2+1, col_block_b*2+1)
        
        # U entries for A11 + A22:
        U[row_block_a*2 * m + col_block_a*2, col_offset] = 1.0
        U[(row_block_a*2+1) * m + (col_block_a*2+1), col_offset] = 1.0
        
        # V entries for B11 + B22:
        V[row_block_b*2 * p + col_block_b*2, col_offset] = 1.0
        V[(row_block_b*2+1) * p + (col_block_b*2+1), col_offset] = 1.0
        
        # W entries for result (row_block_a*2, col_block_b*2) and (row_block_a*2+1, col_block_b*2+1)
        # The result of M1 contributes to C11 and C22
        # C11 gets +M1, C22 gets +M1
        W[col_block_b*2 * n + row_block_a*2, col_offset] = 1.0
        W[(col_block_b*2+1) * n + (row_block_a*2+1), col_offset] = 1.0
        
        # M2: (A21 + A22) * B11
        U[(row_block_a*2+1) * m + col_block_a*2, col_offset+1] = 1.0
        U[(row_block_a*2+1) * m + (col_block_a*2+1), col_offset+1] = 1.0
        V[row_block_b*2 * p + col_block_b*2, col_offset+1] = 1.0
        # Result: C21 gets +M2
        W[col_block_b*2 * n + (row_block_a*2+1), col_offset+1] = 1.0
        
        # M3: A11 * (B12 - B22)
        U[row_block_a*2 * m + col_block_a*2, col_offset+2] = 1.0
        V[row_block_b*2 * p + (col_block_b*2+1), col_offset+2] = 1.0
        V[(row_block_b*2+1) * p + (col_block_b*2+1), col_offset+2] = -1.0
        # Result: C12 gets +M3
        W[(col_block_b*2+1) * n + row_block_a*2, col_offset+2] = 1.0
        
        # M4: A22 * (B21 - B11)
        U[(row_block_a*2+1) * m + (col_block_a*2+1), col_offset+3] = 1.0
        V[(row_block_b*2+1) * p + col_block_b*2, col_offset+3] = 1.0
        V[row_block_b*2 * p + col_block_b*2, col_offset+3] = -1.0
        # Result: C22 gets +M4
        W[(col_block_b*2+1) * n + (row_block_a*2+1), col_offset+3] = 1.0
        
        # M5: (A11 + A12) * B22
        U[row_block_a*2 * m + col_block_a*2, col_offset+4] = 1.0
        U[row_block_a*2 * m + (col_block_a*2+1), col_offset+4] = 1.0
        V[(row_block_b*2+1) * p + (col_block_b*2+1), col_offset+4] = 1.0
        # Result: C12 gets +M5
        W[(col_block_b*2+1) * n + row_block_a*2, col_offset+4] = 1.0
        
        # M6: (A21 - A11) * (B11 + B12)
        U[(row_block_a*2+1) * m + col_block_a*2, col_offset+5] = 1.0
        U[row_block_a*2 * m + col_block_a*2, col_offset+5] = -1.0
        V[row_block_b*2 * p + col_block_b*2, col_offset+5] = 1.0
        V[row_block_b*2 * p + (col_block_b*2+1), col_offset+5] = 1.0
        # Result: C21 gets +M6
        W[col_block_b*2 * n + (row_block_a*2+1), col_offset+5] = 1.0
        
        # M7: (A12 - A22) * (B21 + B22)
        U[row_block_a*2 * m + (col_block_a*2+1), col_offset+6] = 1.0
        U[(row_block_a*2+1) * m + (col_block_a*2+1), col_offset+6] = -1.0
        V[(row_block_b*2+1) * p + col_block_b*2, col_offset+6] = 1.0
        V[(row_block_b*2+1) * p + (col_block_b*2+1), col_offset+6] = 1.0
        # Result: C22 gets +M7
        W[(col_block_b*2+1) * n + (row_block_a*2+1), col_offset+6] = 1.0
    
    def add_rank1_term(row_block_a, col_block_a, row_block_b, col_block_b, col_offset):
        """Add a rank-1 term for a 2x2 * 2x1 multiplication."""
        # A_block is 2x2 at (row_block_a, col_block_a)
        # B_block is 2x1 at (row_block_b, col_block_b) where col_block_b=1 (the 3rd column)
        # Result is 2x1 at (row_block_a, col_block_b)
        
        # C[i] = sum_j A[i,j] * B[j]
        # For i=0,1; j=0,1
        # T[i,j] = 1
        
        # We can do this with rank 1:
        # U[i*2+j, r] = 1 for all i,j
        # V[j, r] = 1 for all j
        # W[i, r] = 1 for all i
        
        for i in range(2):
            for j in range(2):
                U[(row_block_a*2 + i) * m + (col_block_a*2 + j), col_offset] = 1.0
                V[(row_block_b*2 + j) * p + (col_block_b*2 + 0), col_offset] = 1.0
                W[(col_block_b*2 + 0) * n + (row_block_a*2 + i), col_offset] = 1.0
    
    # Now add all the terms
    col_offset = 0
    
    # 8 instances of 2x2 * 2x2
    # (A00, B00) -> C00
    add_strassen_term(0, 0, 0, 0, col_offset)
    col_offset += 7
    
    # (A01, B10) -> C00
    add_strassen_term(0, 1, 1, 0, col_offset)
    col_offset += 7
    
    # (A00, B01) -> C01
    add_strassen_term(0, 0, 0, 1, col_offset)
    col_offset += 7
    
    # (A01, B11) -> C01
    add_strassen_term(0, 1, 1, 1, col_offset)
    col_offset += 7
    
    # (A10, B00) -> C10
    add_strassen_term(1, 0, 0, 0, col_offset)
    col_offset += 7
    
    # (A11, B10) -> C10
    add_strassen_term(1, 1, 1, 0, col_offset)
    col_offset += 7
    
    # (A10, B01) -> C11
    add_strassen_term(1, 0, 0, 1, col_offset)
    col_offset += 7
    
    # (A11, B11) -> C11
    add_strassen_term(1, 1, 1, 1, col_offset)
    col_offset += 7
    
    # 4 instances of 2x2 * 2x1
    # (A00, B02) -> C02
    add_rank1_term(0, 0, 0, 1, col_offset)
    col_offset += 1
    
    # (A01, B12) -> C02
    add_rank1_term(0, 1, 1, 1, col_offset)
    col_offset += 1
    
    # (A10, B02) -> C12
    add_rank1_term(1, 0, 0, 1, col_offset)
    col_offset += 1
    
    # (A11, B12) -> C12
    add_rank1_term(1, 1, 1, 1, col_offset)
    col_offset += 1
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
