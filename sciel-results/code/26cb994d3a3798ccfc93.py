"""
Program for finding a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix.
The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank) = (6, rank)
              V has shape (m*p, rank) = (15, rank)
              W has shape (p*n, rank) = (10, rank)
    """
    n, m, p = DIMENSIONS
    
    # Hybrid Strassen/Schoolbook decomposition
    # Split B (3x5) into B1 (3x2), B2 (3x2), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    # A*B1 and A*B2 use Strassen (rank 7 each), A*B3 uses schoolbook (rank 6)
    # Total rank = 7 + 7 + 6 = 20
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
    
    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
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
    
    def strassen_block(a_row, b_col_offset, w_row_offset, start_col):
        """
        Add Strassen terms for 2x2 * 2x2 block.
        a_row: row index in A (0 or 1)
        b_col_offset: column offset in B (0, 2, or 4)
        w_row_offset: row offset in W (0, 2, or 4)
        start_col: starting column in the factor matrices
        """
        # A indices: a=a_row*3+0, b=a_row*3+1, c=a_row*3+2, d=a_row*3+3? 
        # Wait, A is 2x3. For Strassen 2x2, we use columns 0,1 of A.
        # A = [[a, b, c], [d, e, f]]
        # For Strassen on first 2 columns: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        
        # B is 3x5. For Strassen on columns b_col_offset, b_col_offset+1:
        # B submatrix: [[e, f], [g, h], [i, j]] where e=B[0,b_col_offset], f=B[0,b_col_offset+1], etc.
        
        # Let's define the indices clearly.
        # A indices:
        a_idx = 0 * 3 + 0  # A[0,0]
        b_idx = 0 * 3 + 1  # A[0,1]
        c_idx = 1 * 3 + 0  # A[1,0]
        d_idx = 1 * 3 + 1  # A[1,1]
        
        # B indices:
        e_idx = 0 * 5 + b_col_offset
        f_idx = 0 * 5 + b_col_offset + 1
        g_idx = 1 * 5 + b_col_offset
        h_idx = 1 * 5 + b_col_offset + 1
        
        # W indices: C is 2x5. W[k*n+i, r] where k is col of C, i is row of C.
        # C[0,0] -> W[0*2+0] = W[0]
        # C[0,1] -> W[1*2+0] = W[2]
        # C[1,0] -> W[0*2+1] = W[1]
        # C[1,1] -> W[1*2+1] = W[3]
        
        # For block starting at w_row_offset (which is column offset in C):
        w11_idx = (w_row_offset) * 2 + 0       # C[0, w_row_offset]
        w12_idx = (w_row_offset + 1) * 2 + 0   # C[0, w_row_offset+1]
        w21_idx = (w_row_offset) * 2 + 1       # C[1, w_row_offset]
        w22_idx = (w_row_offset + 1) * 2 + 1   # C[1, w_row_offset+1]
        
        col = start_col
        
        # M1 = (a+d)(e+h)
        add_term(a_idx, e_idx, w11_idx, 1, 1, 1, col)
        add_term(d_idx, e_idx, w11_idx, 1, 1, 1, col)
        add_term(a_idx, h_idx, w11_idx, 1, 1, 1, col)
        add_term(d_idx, h_idx, w11_idx, 1, 1, 1, col)
        # This is wrong. We need to represent (a+d)(e+h) as a single rank-1 term? No, that's not how tensor decomposition works.
        # Each rank-1 term is U[i,r] * V[j,r] * W[k,r].
        # To compute (a+d)(e+h), we need multiple terms or we need to pre-compute sums.
        # But the factor matrices U, V, W are fixed. We cannot pre-compute sums of variables in U or V.
        # Wait, the tensor decomposition is for the multiplication tensor.
        # The multiplication tensor T has entries T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]B[j,k].
        # A rank-r decomposition means T = sum_{r=1}^R u_r v_r w_r.
        # This is equivalent to saying the matrix multiplication can be done with r scalar multiplications.
        # In the Strassen algorithm, we do 7 scalar multiplications.
        # Each scalar multiplication is of the form (linear combination of A) * (linear combination of B).
        # So for each of the 7 products, we have one rank-1 term in the decomposition?
        # No, a single scalar multiplication (sum a_i A_i) * (sum b_j B_j) contributes to multiple C entries.
        # The contribution to C[i,k] from this product is coeff_A[i] * coeff_B[k] * (product).
        # So in the tensor decomposition, for this one "multiplication", we have terms for all (i,k) pairs.
        # Specifically, if M = (sum_i u_i A_i) * (sum_k v_k B_k), then M contributes u_i * v_k * M to C[i,k].
        # In the tensor T = sum_{r} u_r v_r w_r, the r-th term is a rank-1 tensor.
        # If we have 7 such products, do we have 7 rank-1 tensors? No.
        # Each product M_m is a scalar. The tensor for the m-th product is u_m v_m w_m where u_m is the vector of coeffs for A, v_m for B, and w_m is the vector of coeffs for C.
        # Yes! Each scalar multiplication in Strassen corresponds to exactly one rank-1 tensor in the decomposition.
        # So for Strassen 2x2, we have 7 rank-1 tensors.
        
        # Let's redo this.
        # For M1 = (a+d)(e+h):
        # u vector: [1, 0, 1, 0, 0, 0] (for a,b,c,d in A) -> indices 0, 3
        # v vector: [1, 1, 0, 0, 0, 0] (for e,f,g,h in B) -> indices 0, 1
        # w vector: [1, 0, 0, 1] (for C11, C12, C21, C22) -> indices 0, 3 (relative to block)
        
        # Let's implement this properly.
        
        # Clear the previous incorrect logic. We will build the 20 terms explicitly.
        pass

    # Let's just build the 20 terms directly.
    
    # Block 1: A * B1 (B1 is cols 0,1 of B)
    # Strassen terms for 2x2 * 2x2
    # A uses cols 0,1. B uses cols 0,1. C uses cols 0,1.
    
    # M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1. v: B00=1, B11=1. w: C00=1, C11=1.
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # u: A10=1, A11=1. v: B00=1. w: C10=1, C11=1.
    U[3][1] = 1; U[4][1] = 1  # A10 is index 3, A11 is index 4? No.
    # A is 2x3. Indices:
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    # B is 3x5. Indices:
    # B00=0, B01=1, B02=2, B03=3, B04=4
    # B10=5, B11=6, B12=7, B13=8, B14=9
    # B20=10, B21=11, B22=12, B23=13, B24=14
    # C is 2x5. W indices: W[k*n+i]
    # C00=0, C01=2, C02=4, C03=6, C04=8
    # C10=1, C11=3, C12=5, C13=7, C14=9
    
    # Let's restart with correct indices.
    
    # Reset matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Block 1: A * B1 (B1 = B[:, 0:2]) ---
    # Strassen for A[:, 0:2] * B[:, 0:2]
    # A sub: [[A00, A01], [A10, A11]] -> indices 0, 1, 3, 4
    # B sub: [[B00, B01], [B10, B11]] -> indices 0, 1, 5, 6
    # C sub: [[C00, C01], [C10, C11]] -> indices 0, 2, 1, 3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # --- Block 2: A * B2 (B2 = B[:, 2:4]) ---
    # Strassen for A[:, 0:2] * B[:, 2:4]
    # A sub: same as before
    # B sub: [[B02, B03], [B12, B13]] -> indices 2, 3, 7, 8
    # C sub: [[C02, C03], [C12, C13]] -> indices 4, 6, 5, 7
    
    # M8 = (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M9 = (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # M10 = A00(B03-B13)
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M11 = A11(B12-B02)
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[7][col] = 1
    col += 1
    
    # M12 = (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[4][col] = 1; W[6][col] = 1
    col += 1
    
    # M13 = (A10-A00)(B02+B03)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M14 = (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # --- Block 3: A * B3 (B3 = B[:, 4]) ---
    # Schoolbook for A * B3 (2x3 * 3x1)
    # C04 = A00*B04 + A01*B14 + A02*B24
    # C14 = A10*B04 + A11*B14 + A12*B24
    
    # M15 = A00*B04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # M16 = A01*B14
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # M17 = A02*B24
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    # M18 = A10*B04
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # M19 = A11*B14
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # M20 = A12*B24
    U[5][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
