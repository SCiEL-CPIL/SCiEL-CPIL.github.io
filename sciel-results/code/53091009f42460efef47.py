"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = A22 * (B12 - B22)
    # M4 = A11 * (B21 + B22)
    # M5 = (A12 - A11) * (B11 + B21)
    # M6 = (A21 + A11) * (B12 - B22)
    # M7 = (A22 - A21) * (B11 + B21)
    
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Factor definitions: (A_indices, B_indices, C_coefficients)
    # We will map these to the global indices.
    
    def add_strassen_block(a_offset, b_offset, c_offset, col_start):
        """
        Adds a 2x2x2 Strassen decomposition to the global matrices.
        a_offset: row offset in A (0 or 2)
        b_offset: row offset in B (0, 2, or 4? No, B is 4x6. 
                  B_T is rows 0-1, B_B is rows 2-3.
                  Within B_T, we split cols into 3 blocks of 2.
                  So b_offset is the row offset in B (0 for top, 2 for bottom).
                  c_offset: col offset in C (0, 2, 4)
        """
        # A indices relative to a_offset
        # A is 2x4. a_offset is 0 (cols 0-1) or 2 (cols 2-3).
        # A indices in U: row i*m + j. m=4.
        # So A[0,0] is U[0], A[0,1] is U[1], A[1,0] is U[4], A[1,1] is U[5].
        # Wait, U index is i*m + j.
        # A_L (cols 0,1): i=0,1; j=0,1.
        #   A[0,0] -> U[0*4+0] = U[0]
        #   A[0,1] -> U[0*4+1] = U[1]
        #   A[1,0] -> U[1*4+0] = U[4]
        #   A[1,1] -> U[1*4+1] = U[5]
        
        # A_R (cols 2,3): i=0,1; j=2,3.
        #   A[0,2] -> U[0*4+2] = U[2]
        #   A[0,3] -> U[0*4+3] = U[3]
        #   A[1,2] -> U[1*4+2] = U[6]
        #   A[1,3] -> U[1*4+3] = U[7]
        
        # B indices in V: row j*p + k. p=6.
        # B_T (rows 0,1): j=0,1.
        #   B[0,0] -> V[0*6+0] = V[0]
        #   B[0,1] -> V[0*6+1] = V[1]
        #   B[1,0] -> V[1*6+0] = V[6]
        #   B[1,1] -> V[1*6+1] = V[7]
        
        # B_B (rows 2,3): j=2,3.
        #   B[2,0] -> V[2*6+0] = V[12]
        #   B[2,1] -> V[2*6+1] = V[13]
        #   B[3,0] -> V[3*6+0] = V[18]
        #   B[3,1] -> V[3*6+1] = V[19]
        
        # C indices in W: row k*n + i. n=2.
        # C block at col c_offset.
        # C[0,0] -> W[0*2+0] = W[0]
        # C[0,1] -> W[1*2+0] = W[2]
        # C[1,0] -> W[0*2+1] = W[1]
        # C[1,1] -> W[1*2+1] = W[3]
        
        # Wait, W index is k*n + i.
        # k is col of C (0..5), i is row of C (0..1).
        # For a 2x2 C block starting at col c_offset:
        # C[0, c_offset] -> W[c_offset*2 + 0]
        # C[0, c_offset+1] -> W[(c_offset+1)*2 + 0]
        # C[1, c_offset] -> W[c_offset*2 + 1]
        # C[1, c_offset+1] -> W[(c_offset+1)*2 + 1]
        
        # Strassen Factors
        # M1 = A11 * (B11 + B22)
        # A11: a_offset==0 ? U[0] : U[2]
        # B11: b_offset==0 ? V[0] : V[12] (if b_offset is row offset)
        #      Actually b_offset is the start row of the 2x2 block in B.
        #      If b_offset=0, B11 is V[0], B22 is V[7].
        #      If b_offset=2, B11 is V[12], B22 is V[19].
        
        a11_idx = (0 * 4 + a_offset) # i=0, j=a_offset
        a12_idx = (0 * 4 + a_offset + 1)
        a21_idx = (1 * 4 + a_offset)
        a22_idx = (1 * 4 + a_offset + 1)
        
        b11_idx = (b_offset * 6 + c_offset) # j=b_offset, k=c_offset
        b12_idx = (b_offset * 6 + c_offset + 1)
        b21_idx = ((b_offset + 1) * 6 + c_offset)
        b22_idx = ((b_offset + 1) * 6 + c_offset + 1)
        
        c11_idx = (c_offset * 2 + 0) # k=c_offset, i=0
        c12_idx = ((c_offset + 1) * 2 + 0)
        c21_idx = (c_offset * 2 + 1)
        c22_idx = ((c_offset + 1) * 2 + 1)
        
        # M1
        U[a11_idx][col_start] = 1
        V[b11_idx][col_start] = 1
        V[b22_idx][col_start] = 1
        W[c11_idx][col_start] = 1
        W[c22_idx][col_start] = 1
        
        # M2
        U[a11_idx][col_start+1] = 1
        U[a22_idx][col_start+1] = 1
        V[b22_idx][col_start+1] = 1
        W[c21_idx][col_start+1] = 1
        W[c22_idx][col_start+1] = -1
        
        # M3
        U[a22_idx][col_start+2] = 1
        V[b12_idx][col_start+2] = 1
        V[b22_idx][col_start+2] = -1
        W[c11_idx][col_start+2] = 1
        W[c22_idx][col_start+2] = 1
        
        # M4
        U[a11_idx][col_start+3] = 1
        V[b21_idx][col_start+3] = 1
        V[b22_idx][col_start+3] = 1
        W[c11_idx][col_start+3] = -1
        W[c12_idx][col_start+3] = 1
        
        # M5
        U[a12_idx][col_start+4] = 1
        U[a11_idx][col_start+4] = -1
        V[b11_idx][col_start+4] = 1
        V[b21_idx][col_start+4] = 1
        W[c12_idx][col_start+4] = 1
        W[c22_idx][col_start+4] = 1
        
        # M6
        U[a21_idx][col_start+5] = 1
        U[a11_idx][col_start+5] = 1
        V[b12_idx][col_start+5] = 1
        V[b22_idx][col_start+5] = -1
        W[c21_idx][col_start+5] = 1
        
        # M7
        U[a22_idx][col_start+6] = 1
        U[a21_idx][col_start+6] = -1
        V[b11_idx][col_start+6] = 1
        V[b21_idx][col_start+6] = 1
        W[c11_idx][col_start+6] = 1

    # We have 2 blocks of A (offsets 0, 2) and 3 blocks of B cols (offsets 0, 2, 4)
    # But we split B into Top (rows 0-1) and Bottom (rows 2-3).
    # So for each A block, we have 2 B row blocks.
    # And for each B row block, we have 3 B col blocks.
    # Total blocks: 2 (A) * 2 (B rows) * 3 (B cols) = 12 blocks of 2x2x2.
    # Each block uses 7 columns. 12 * 7 = 84?
    # Wait. My previous logic was:
    # C = A_L B_T + A_R B_B.
    # A_L B_T is 2x2x6. Rank 21.
    # A_R B_B is 2x2x6. Rank 21.
    # Total 42.
    
    # So for A_L (a_offset=0):
    #   B_T (b_offset=0): c_offsets 0, 2, 4. (3 blocks)
    #   B_B (b_offset=2): c_offsets 0, 2, 4. (3 blocks)
    #   Total 6 blocks of 2x2x2. 6 * 7 = 42?
    #   No. A_L B_T + A_L B_B?
    #   C = A_L B_T + A_R B_B.
    #   A_L is 2x2. B_T is 2x6.
    #   A_L B_T requires 21 multiplications.
    #   A_R B_B requires 21 multiplications.
    #   Total 42.
    
    # So:
    # 1. A_L (a_offset=0) with B_T (b_offset=0). c_offsets 0, 2, 4.
    #    This covers C = A_L * B_T.
    #    But B_T is only the top half of B.
    #    Wait. B is 4x6.
    #    A_L is cols 0-1 of A.
    #    A_L multiplies the top half of B (rows 0-1) to contribute to C?
    #    Yes, C = A B.
    #    C[i, k] = sum_j A[i, j] B[j, k].
    #    A_L has non-zeros in j=0,1.
    #    So A_L multiplies B[0:2, :] (Top half of B).
    #    A_R has non-zeros in j=2,3.
    #    So A_R multiplies B[2:4, :] (Bottom half of B).
    
    # So:
    # Term 1: A_L * B_T. (2x2) * (2x6).
    # Term 2: A_R * B_B. (2x2) * (2x6).
    
    # For Term 1:
    #   a_offset = 0.
    #   b_offset = 0.
    #   c_offsets = 0, 2, 4.
    #   3 blocks of Strassen.
    
    # For Term 2:
    #   a_offset = 2.
    #   b_offset = 2.
    #   c_offsets = 0, 2, 4.
    #   3 blocks of Strassen.
    
    # Total 6 blocks. 6 * 7 = 42.
    
    col = 0
    # Term 1: A_L * B_T
    for c_off in [0, 2, 4]:
        add_strassen_block(0, 0, c_off, col)
        col += 7
        
    # Term 2: A_R * B_B
    for c_off in [0, 2, 4]:
        add_strassen_block(2, 2, c_off, col)
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
