"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 78 decomposition
    # Split A (3x4) into A_L (3x2) and A_R (3x2)
    # Split B (4x7) into B_L (2x7) and B_R (2x7)
    # C = A_L B_L + A_R B_R
    
    # For each half (L and R), we compute 3x2x7
    # Split A_half (3x2) into A_h1 (1x2) and A_h2 (2x2)
    # C_h = A_h1 B_half + A_h2 B_half
    # A_h1 B_half is 1x2x7 (rank 14)
    # A_h2 B_half is 2x2x7
    # Split B_half (2x7) into B_h1 (2x2), B_h2 (2x2), B_h3 (2x2), B_h4 (2x1)
    # A_h2 B_h1 is 2x2x2 (rank 7, Strassen)
    # A_h2 B_h2 is 2x2x2 (rank 7, Strassen)
    # A_h2 B_h3 is 2x2x2 (rank 7, Strassen)
    # A_h2 B_h4 is 2x2x1 (rank 4, schoolbook)
    # Total per half: 14 + 7 + 7 + 7 + 4 = 39
    # Total rank: 39 + 39 = 78
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)] # 12 x 78
    V = [[0] * rank for _ in range(m * p)] # 28 x 78
    W = [[0] * rank for _ in range(p * n)] # 21 x 78
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    def strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds Strassen terms for a 2x2x2 block.
        A is at rows a_row_offset, a_row_offset+1, cols 0,1 (relative to A_half)
        B is at rows 0,1, cols b_col_offset, b_col_offset+1 (relative to B_half)
        C is at rows c_row_offset, c_row_offset+1, cols c_col_offset, c_col_offset+1 (relative to C_half)
        
        Note: The indices for U, V, W need to map to the global A, B, C.
        A_half is 3x2. A_h2 is the bottom 2 rows of A_half.
        So A_h2 rows are 1, 2 of A_half.
        B_half is 2x7.
        C_half is 3x7.
        
        This function is too complex to inline easily without careful index mapping.
        I will write a helper to get global indices.
        """
        pass

    # Let's do it manually for clarity and correctness.
    
    # --- Part L ---
    # A_L is rows 0,1,2 of A, cols 0,1 of A.
    # B_L is rows 0,1 of B, cols 0,6 of B.
    # C_L is rows 0,1,2 of C, cols 0,6 of C.
    
    # A_L1 (1x2): Row 0 of A_L. Cols 0,1.
    # B_L (2x7): Rows 0,1. Cols 0,6.
    # C_L1 (1x7): Row 0 of C_L. Cols 0,6.
    # Terms: a_00*b_0j + a_01*b_1j
    for j in range(7):
        # Term 1: a_00 * b_0j
        add_term(0, j, j, 1, 1, 1) # U[0,0]=1, V[0,j]=1, W[j,0]=1
        # Term 2: a_01 * b_1j
        add_term(1, 7+j, j, 1, 1, 1) # U[0,1]=1, V[1,j]=1, W[j,0]=1
        
    # A_L2 (2x2): Rows 1,2 of A_L. Cols 0,1.
    # B_L1 (2x2): Rows 0,1 of B_L. Cols 0,1.
    # C_L21 (2x2): Rows 1,2 of C_L. Cols 0,1.
    # Strassen for A_L2 * B_L1
    # A_L2 indices in U: (1,0), (1,1), (2,0), (2,1) -> U indices 2,3,4,5? No.
    # U row index = i*m + j.
    # A_L is cols 0,1. So j=0,1.
    # A_L2 rows are i=1,2.
    # U indices:
    # a11 (i=1,j=0): 1*4+0 = 4
    # a12 (i=1,j=1): 1*4+1 = 5
    # a21 (i=2,j=0): 2*4+0 = 8
    # a22 (i=2,j=1): 2*4+1 = 9
    
    # B_L1 indices in V:
    # B_L is rows 0,1. Cols 0,1.
    # V row index = j*p + k.
    # b11 (j=0,k=0): 0*7+0 = 0
    # b12 (j=0,k=1): 0*7+1 = 1
    # b21 (j=1,k=0): 1*7+0 = 7
    # b22 (j=1,k=1): 1*7+1 = 8
    
    # C_L21 indices in W:
    # C_L is rows 1,2. Cols 0,1.
    # W row index = k*n + i.
    # c11 (i=1,k=0): 0*3+1 = 1
    # c12 (i=1,k=1): 1*3+1 = 4
    # c21 (i=2,k=0): 0*3+2 = 2
    # c22 (i=2,k=1): 1*3+2 = 5
    
    # Strassen M1..M7
    # M1 = (a11+a22)(b11+b22)
    # U: a11(4), a22(9). V: b11(0), b22(8). W: c11(1), c22(5).
    # We need to add terms for each product.
    # This is getting complex. I will write a generic helper.
    
    def add_strassen(u_a11, u_a12, u_a21, u_a22, 
                     v_b11, v_b12, v_b21, v_b22, 
                     w_c11, w_c12, w_c21, w_c22):
        nonlocal col
        # M1 = (a11+a22)(b11+b22)
        add_term(u_a11, v_b11, w_c11, 1, 1, 1)
        add_term(u_a11, v_b22, w_c11, 1, 1, 1)
        add_term(u_a22, v_b11, w_c11, 1, 1, 1)
        add_term(u_a22, v_b22, w_c11, 1, 1, 1)
        # Wait, Strassen uses intermediate sums.
        # The tensor decomposition requires rank-1 terms.
        # M1 is a product of sums. It expands to 4 rank-1 terms.
        # But Strassen replaces 8 terms with 7.
        # The 7 terms are:
        # 1: (a11+a22)(b11+b22) -> 4 terms in tensor? No.
        # The decomposition is:
        # C = sum M_k * (u_k \otimes v_k \otimes w_k)
        # Where M_k are the linear combinations.
        # So we add 7 terms.
        
        # Term 1: (a11+a22) * (b11+b22)
        # This is not a single rank-1 term.
        # The rank-1 terms are the products of the linear combinations.
        # So we add:
        # 1. (a11+a22) (b11+b22) -> contributes to c11, c22
        # This requires adding 4 entries to U, V, W?
        # No, the rank-1 term is defined by vectors u, v, w.
        # u has non-zero entries at a11, a22.
        # v has non-zero entries at b11, b22.
        # w has non-zero entries at c11, c22.
        
        # So:
        # Term 1: u=[a11,a22], v=[b11,b22], w=[c11,c22]
        U[u_a11][col] = 1
        U[u_a22][col] = 1
        V[v_b11][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1
        
        # Term 2: (a21+a22) * b11
        U[u_a21][col] = 1
        U[u_a22][col] = 1
        V[v_b11][col] = 1
        W[w_c21][col] = 1
        W[w_c22][col] = 1 # Wait, c21 and c22?
        # M2 = (a21+a22)b11.
        # c21 += M2, c22 += M2?
        # Strassen:
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        # M2 = (a21+a22)b11.
        # So w has entries at c21, c22.
        W[w_c21][col] = 1
        W[w_c22][col] = 1
        col += 1

        # Term 3: a11 * (b12-b22)
        U[u_a11][col] = 1
        V[v_b12][col] = 1
        V[v_b22][col] = -1
        W[w_c12][col] = 1
        W[w_c22][col] = 1
        col += 1

        # Term 4: a22 * (b21-b11)
        U[u_a22][col] = 1
        V[v_b21][col] = 1
        V[v_b11][col] = -1
        W[w_c11][col] = 1
        W[w_c21][col] = 1
        col += 1

        # Term 5: (a11+a12) * b22
        U[u_a11][col] = 1
        U[u_a12][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = -1
        W[w_c12][col] = 1
        col += 1

        # Term 6: (a21-a11) * (b11+b12)
        U[u_a21][col] = 1
        U[u_a11][col] = -1
        V[v_b11][col] = 1
        V[v_b12][col] = 1
        W[w_c22][col] = 1
        col += 1

        # Term 7: (a12-a22) * (b21+b22)
        U[u_a12][col] = 1
        U[u_a22][col] = -1
        V[v_b21][col] = 1
        V[v_b22][col] = 1
        W[w_c11][col] = 1
        W[w_c22][col] = 1
        col += 1

    # Call add_strassen for B_L1
    add_strassen(4, 5, 8, 9, 0, 1, 7, 8, 1, 4, 2, 5)
    
    # B_L2 (2x2): Cols 2,3
    add_strassen(4, 5, 8, 9, 2, 3, 9, 10, 1, 5, 2, 6)
    
    # B_L3 (2x2): Cols 4,5
    add_strassen(4, 5, 8, 9, 4, 5, 11, 12, 1, 6, 2, 7)
    
    # B_L4 (2x1): Col 6
    # Schoolbook 2x2x1
    # a11*b11, a12*b21, a21*b11, a22*b21
    # b11 is V[6], b21 is V[13]
    # c11 is W[7], c21 is W[10]
    add_term(4, 6, 7, 1, 1, 1)
    add_term(5, 13, 7, 1, 1, 1)
    add_term(8, 6, 10, 1, 1, 1)
    add_term(9, 13, 10, 1, 1, 1)
    
    # --- Part R ---
    # A_R is rows 0,1,2 of A, cols 2,3 of A.
    # B_R is rows 2,3 of B, cols 0,6 of B.
    # C_R is rows 0,1,2 of C, cols 0,6 of C.
    
    # A_R1 (1x2): Row 0 of A_R. Cols 2,3.
    # U indices:
    # a03 (i=0,j=2): 0*4+2 = 2
    # a04 (i=0,j=3): 0*4+3 = 3
    for j in range(7):
        # Term 1: a03 * b2j (B_R row 0 is global row 2)
        # V index for B_R row 0, col j: 2*7+j = 14+j
        add_term(2, 14+j, j, 1, 1, 1)
        # Term 2: a04 * b3j (B_R row 1 is global row 3)
        # V index for B_R row 1, col j: 3*7+j = 21+j
        add_term(3, 21+j, j, 1, 1, 1)
        
    # A_R2 (2x2): Rows 1,2 of A_R. Cols 2,3.
    # U indices:
    # a13 (i=1,j=2): 1*4+2 = 6
    # a14 (i=1,j=3): 1*4+3 = 7
    # a23 (i=2,j=2): 2*4+2 = 10
    # a24 (i=2,j=3): 2*4+3 = 11
    
    # B_R1 (2x2): Rows 2,3 of B. Cols 0,1.
    # V indices:
    # b21 (j=2,k=0): 2*7+0 = 14
    # b22 (j=2,k=1): 2*7+1 = 15
    # b31 (j=3,k=0): 3*7+0 = 21
    # b32 (j=3,k=1): 3*7+1 = 22
    
    # C_R21 (2x2): Rows 1,2 of C. Cols 0,1.
    # W indices:
    # c11 (i=1,k=0): 0*3+1 = 1
    # c12 (i=1,k=1): 1*3+1 = 4
    # c21 (i=2,k=0): 0*3+2 = 2
    # c22 (i=2,k=1): 1*3+2 = 5
    
    add_strassen(6, 7, 10, 11, 14, 15, 21, 22, 1, 4, 2, 5)
    
    # B_R2 (2x2): Cols 2,3
    add_strassen(6, 7, 10, 11, 16, 17, 23, 24, 1, 5, 2, 6)
    
    # B_R3 (2x2): Cols 4,5
    add_strassen(6, 7, 10, 11, 18, 19, 25, 26, 1, 6, 2, 7)
    
    # B_R4 (2x1): Col 6
    # V indices: b24 (14+6=20), b34 (21+6=27)
    # W indices: c13 (7), c23 (10)
    add_term(6, 20, 7, 1, 1, 1)
    add_term(7, 27, 7, 1, 1, 1)
    add_term(10, 20, 10, 1, 1, 1)
    add_term(11, 27, 10, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
