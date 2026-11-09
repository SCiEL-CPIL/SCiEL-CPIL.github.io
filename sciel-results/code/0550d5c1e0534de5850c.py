"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # We want to decompose the matrix multiplication tensor for (n,m,p) = (4,4,7)
    # T[i,j,k] = 1 for the standard matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Use Strassen for the 4x4 x 4x4 part (columns 0-3 of B)
    # and schoolbook for the 4x4 x 3 part (columns 4-6 of B)
    
    # For the 4x4 x 4x4 part:
    # Split into 2x2 blocks and use Strassen (7 multiplications of 2x2)
    # Each 2x2 multiplication has rank 4 (schoolbook) or can use a lower-rank decomposition
    # Actually, for 2x2 matrix multiplication, the minimum rank is 7 (Strassen)
    # But we're doing 4x4 x 4x4, not 2x2 x 2x2
    
    # Let me think about this more carefully.
    # 4x4 x 4x4 matrix multiplication can be done with:
    # - Schoolbook: 4*4*4 = 64 scalar multiplications
    # - Strassen: 7 * (2x2 mult) = 7 * 4 = 28 scalar multiplications (if 2x2 mult has rank 4)
    #   But 2x2 mult actually has rank 7, so Strassen gives 7*7 = 49? No...
    
    # Actually, the rank of 2x2 matrix multiplication is 7 (Strassen).
    # So 4x4 x 4x4 via Strassen: split into 2x2 blocks, 7 block multiplications,
    # each 2x2 mult has rank 7, so total rank = 7 * 7 = 49
    
    # For 4x4 x 4x3:
    # Schoolbook: 4*4*3 = 48 scalar multiplications
    # Can we do better? Maybe split the 3 columns...
    # 4x4 x 4x3 is not a standard square problem.
    # We could treat it as 4x4 x 4x4 with 3 columns zero, but that doesn't help.
    # Or we could use a different decomposition.
    
    # Total rank with Strassen for 4x4x4 + schoolbook for 4x4x3:
    # 49 + 48 = 97
    
    # Alternative: Use Strassen for the full 4x4 x 4x7?
    # That's not directly applicable since Strassen works for square matrices.
    
    # Let me try a different approach: 
    # Use the fact that 4x4 x 4x7 can be decomposed as:
    # C = A * B where A is 4x4 and B is 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # Compute A*B1 using Strassen (rank 49)
    # Compute A*B2 using schoolbook (rank 48)
    # Total: 97
    
    # Can we do better for A*B2?
    # 4x4 x 4x3: we can view this as computing 3 columns of a 4x4 product
    # Each column is a 4x1 vector = A * (4x1 vector)
    # 4x1 vector = 4*1 = 4 scalar mults per column, 3 columns = 12 scalar mults
    # Wait, that's not right. A*B2 where B2 is 4x3:
    # C2[i,k] = sum_j A[i,j]*B2[j,k] for i=0..3, k=0..2
    # That's 4*3*4 = 48 scalar mults for schoolbook.
    
    # Hmm, can we use a fast algorithm for 4x4 x 4x3?
    # One approach: pad B2 to 4x4 with zeros, compute 4x4x4 with Strassen (rank 49),
    # but that gives rank 49 for the 4x3 part, worse than schoolbook 48.
    
    # Another approach: use the 4x4x4 Strassen decomposition but only keep
    # the terms that contribute to the first 3 columns.
    
    # Let me just go with Strassen for 4x4x4 + schoolbook for 4x4x3 = 97 total.
    # That's better than schoolbook 112.
    
    # Now I need to actually construct the tensor decomposition.
    
    # For the 4x4x4 Strassen part:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B1 = [[B11, B12], [B21, B22]] where each block is 2x2
    # C1 = A*B1 = [[C11, C12], [C21, C22]]
    
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which has rank 7 (Strassen for 2x2).
    # But I need to decompose each 2x2 product into rank-1 terms.
    
    # For 2x2 matrix multiplication, Strassen gives:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # But each Pi is a scalar product of two scalars? No, Pi is a scalar
    # (product of two 1x1 matrices, i.e., scalars).
    
    # Wait, I'm confusing levels. Let me be very explicit.
    
    # At the top level (4x4 x 4x4):
    # We split into 2x2 blocks. Each "block multiplication" like A11*B11
    # is a 2x2 x 2x2 matrix multiplication, which itself can be done with
    # Strassen (rank 7) or schoolbook (rank 4).
    
    # If I use schoolbook for each 2x2 block multiplication:
    # There are 7 block multiplications in Strassen, each with rank 4
    # Total rank = 7 * 4 = 28 for the 4x4x4 part
    
    # If I use Strassen for each 2x2 block multiplication:
    # There are 7 block multiplications, each with rank 7
    # Total rank = 7 * 7 = 49 for the 4x4x4 part
    
    # So using schoolbook for the inner 2x2 multiplications is better!
    # Total for 4x4x4: 28
    # Total for 4x4x3: 48
    # Grand total: 28 + 48 = 76
    
    # Let me verify: 4x4 x 4x4 with outer Strassen + inner schoolbook:
    # 7 outer multiplications, each is a 2x2x2x2 schoolbook = 4 scalar mults
    # 7 * 4 = 28. Yes!
    
    # So the plan:
    # 1. For columns 0-3 of B (the 4x4 part): use Strassen with schoolbook inner
    #    Rank = 28
    # 2. For columns 4-6 of B (the 4x3 part): use schoolbook
    #    Rank = 48
    # Total rank = 76
    
    # Now let me construct the actual U, V, W matrices.
    
    rank = 76
    U = np.zeros((n * m, rank))  # (16, 76)
    V = np.zeros((m * p, rank))  # (28, 76)
    W = np.zeros((p * n, rank))  # (28, 76)
    
    col = 0
    
    # Part 1: Strassen for 4x4 x 4x4 (columns 0-3)
    # Block definitions:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product computed with schoolbook (4 scalar mults)
    # Mi[di,dk] = sum_dj X[di,dj] * Y[dj,dk] where X,Y are 2x2 matrices
    
    # For each Mi, I need to add 4 rank-1 terms to U, V, W
    
    def add_2x2_mult(X_coeffs, Y_coeffs, C_offset_row, C_offset_col, c_coeff):
        """
        Add terms for a 2x2 matrix multiplication X*Y -> C_block
        X_coeffs: 2x2 array of coefficients for X[di,dj]
                  X[di,dj] = sum over (i,j) of coeff * A[i,j]
                  The (i,j) pair determines the U index: i*m + j
                  The coefficient is the U value
        Y_coeffs: 2x2 array of coefficients for Y[dj,dk]
                  Y[dj,dk] = sum over (j,k) of coeff * B[j,k]
                  The (j,k) pair determines the V index: j*p + k
                  The coefficient is the V value
        C_offset_row, C_offset_col: offset in C for the result block
        c_coeff: coefficient for the W matrix
        Returns the number of columns used
        """
        nonlocal col
        # X[di,dj] contributes to C[di,dk] via sum_dj X[di,dj]*Y[dj,dk]
        # Each term: X[di,dj] * Y[dj,dk] -> C[C_offset_row+di, C_offset_col+dk]
        
        # X[di,dj] is a linear combination of A[i,j] entries
        # Y[dj,dk] is a linear combination of B[j,k] entries
        # The product X[di,dj]*Y[dj,dk] is a sum of products A[i,j]*B[j,k]
        
        # For schoolbook 2x2 mult:
        # C[di,dk] = sum_{dj=0}^{1} X[di,dj] * Y[dj,dk]
        # Each X[di,dj]*Y[dj,dk] is one "multiplication"
        
        # But X[di,dj] itself may be a sum of A entries,
        # and Y[dj,dk] may be a sum of B entries.
        # So X[di,dj]*Y[dj,dk] = sum_{(i,j) in supp(X[di,dj])} sum_{(j',k) in supp(Y[dj,dk])}
        #   coeff_x * coeff_y * A[i,j] * B[j',k]
        # For this to be valid, we need j = j' (the inner index must match)
        
        # Actually, in the Strassen construction, X and Y are specific 2x2 matrices
        # whose entries are specific A or B entries (or sums/differences).
        # The key constraint is that the inner index j must match.
        
        # Let me define X and Y as dictionaries mapping (di,dj) to a list of
        # (i, j, coeff) triples, and similarly for Y.
        
        # For a valid 2x2 matrix multiplication X*Y:
        # C[di,dk] = sum_dj X[di,dj] * Y[dj,dk]
        # where X[di,dj] is a linear combo of A entries with fixed i=di+offset
        # and Y[dj,dk] is a linear combo of B entries with fixed j=dj+offset
        # The constraint is that the j index in X[di,dj] matches the j index in Y[dj,dk]
        
        # This is getting complicated. Let me just hardcode the terms.
        
        # For each (di, dj, dk) in {0,1}^3:
        #   X[di,dj] * Y[dj,dk] contributes to C[C_offset_row+di, C_offset_col+dk]
        #   X[di,dj] = sum of A[i,j] terms
        #   Y[dj,dk] = sum of B[j,k] terms
        #   Product = sum of A[i,j]*B[j,k] terms (with j matching)
        
        # I'll represent X[di,dj] as a list of (i, j, coeff) and
        # Y[dj,dk] as a list of (j, k, coeff)
        
        # For each (di, dk), C[di,dk] = sum_dj X[di,dj]*Y[dj,dk]
        # For each dj, X[di,dj]*Y[dj,dk] = sum over matching j of products
        
        # Let me just enumerate all terms directly.
        
        for di in range(2):
            for dk in range(2):
                for dj in range(2):
                    # X[di,dj] * Y[dj,dk]
                    # X[di,dj] is represented as a list of (i, j, coeff)
                    # Y[dj,dk] is represented as a list of (j, k, coeff)
                    # For each (i, j, cx) in X[di,dj] and (j', k, cy) in Y[dj,dk]:
                    #   if j == j': add term A[i,j]*B[j,k] -> C[C_offset_row+di, C_offset_col+dk]
                    #   with U coeff = cx, V coeff = cy, W coeff = c_coeff
                    
                    for (i, j, cx) in X_coeffs[di, dj]:
                        for (j2, k, cy) in Y_coeffs[dj, dk]:
                            if j == j2:
                                U[i * m + j, col] += cx
                                V[j * p + k, col] += cy
                                W[(C_offset_col + dk) * n + (C_offset_row + di), col] += c_coeff
        col += 1
        return 1
    
    # Hmm, this approach is getting unwieldy. Let me take a different tactic.
    
    # I'll directly enumerate all the scalar-level terms for the Strassen decomposition.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # For the Strassen part (4x4 x 4x4, columns 0-3 of B):
    # I'll define each of the 7 Strassen multiplications as a set of scalar terms.
    
    # M1 = (A11+A22)*(B11+B22)
    # A11+A22 is a 2x2 matrix:
    #   (A11+A22)[0,0] = A[0,0] + A[2,2]
    #   (A11+A22)[0,1] = A[0,1] + A[2,3]
    #   (A11+A22)[1,0] = A[1,0] + A[3,2]
    #   (A11+A22)[1,1] = A[1,1] + A[3,3]
    # B11+B22 is a 2x2 matrix:
    #   (B11+B22)[0,0] = B[0,0] + B[2,2]
    #   (B11+B22)[0,1] = B[0,1] + B[2,3]
    #   (B11+B22)[1,0] = B[1,0] + B[3,2]
    #   (B11+B22)[1,1] = B[1,1] + B[3,3]
    
    # M1[di,dk] = sum_dj (A11+A22)[di,dj] * (B11+B22)[dj,dk]
    
    # M1 contributes +1 to C11 (offset 0,0) and +1 to C22 (offset 2,2)
    
    # Let me define a helper that adds all terms for a 2x2 product of two
    # "expanded" 2x2 matrices, contributing to a C block with a given coefficient.
    
    def add_strassen_term(A_expanded, B_expanded, C_row_offset, C_col_offset, c_coeff):
        """
        A_expanded: 2x2 array where A_expanded[di,dj] is a list of (i, j, coeff)
        B_expanded: 2x2 array where B_expanded[dj,dk] is a list of (j, k, coeff)
        C_row_offset, C_col_offset: offset in C
        c_coeff: coefficient for W
        """
        nonlocal col
        for di in range(2):
            for dk in range(2):
                for dj in range(2):
                    for (i, j, ca) in A_expanded[di, dj]:
                        for (j2, k, cb) in B_expanded[dj, dk]:
                            if j == j2:
                                U[i * m + j, col] += ca
                                V[j * p + k, col] += cb
                                W[(C_col_offset + dk) * n + (C_row_offset + di), col] += c_coeff
        col += 1
    
    # Define the expanded matrices for each Strassen multiplication
    
    # M1 = (A11+A22)*(B11+B22)
    A_M1 = np.empty((2, 2), dtype=object)
    B_M1 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M1[di, dj] = [(di, dj, 1), (di+2, dj+2, 1)]
            B_M1[dj, dk if False else dj, ] if False else None
    # Fix B_M1
    for dj in range(2):
        for dk in range(2):
            B_M1[dj, dk] = [(dj, dk, 1), (dj+2, dk+2, 1)]
    
    # M1 contributes to C11 (offset 0,0) with +1 and C22 (offset 2,2) with +1
    add_strassen_term(A_M1, B_M1, 0, 0, 1)  # C11
    add_strassen_term(A_M1, B_M1, 2, 2, 1)  # C22
    
    # M2 = (A21+A22)*B11
    A_M2 = np.empty((2, 2), dtype=object)
    B_M2 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M2[di, dj] = [(di+2, dj, 1), (di+2, dj+2, 1)]
    for dj in range(2):
        for dk in range(2):
            B_M2[dj, dk] = [(dj, dk, 1)]
    
    # M2 contributes to C21 (offset 2,0) with +1 and C22 (offset 2,2) with -1
    add_strassen_term(A_M2, B_M2, 2, 0, 1)  # C21
    add_strassen_term(A_M2, B_M2, 2, 2, -1)  # C22
    
    # M3 = A11*(B12-B22)
    A_M3 = np.empty((2, 2), dtype=object)
    B_M3 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M3[di, dj] = [(di, dj, 1)]
    for dj in range(2):
        for dk in range(2):
            B_M3[dj, dk] = [(dj, dk+2, 1), (dj+2, dk+2, -1)]
    
    # M3 contributes to C12 (offset 0,2) with +1 and C22 (offset 2,2) with +1
    add_strassen_term(A_M3, B_M3, 0, 2, 1)  # C12
    add_strassen_term(A_M3, B_M3, 2, 2, 1)  # C22
    
    # M4 = A22*(B21-B11)
    A_M4 = np.empty((2, 2), dtype=object)
    B_M4 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M4[di, dj] = [(di+2, dj+2, 1)]
    for dj in range(2):
        for dk in range(2):
            B_M4[dj, dk] = [(dj+2, dk, 1), (dj, dk, -1)]
    
    # M4 contributes to C11 (offset 0,0) with +1 and C21 (offset 2,0) with +1
    add_strassen_term(A_M4, B_M4, 0, 0, 1)  # C11
    add_strassen_term(A_M4, B_M4, 2, 0, 1)  # C21
    
    # M5 = (A11+A12)*B22
    A_M5 = np.empty((2, 2), dtype=object)
    B_M5 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M5[di, dj] = [(di, dj, 1), (di, dj+2, 1)]
    for dj in range(2):
        for dk in range(2):
            B_M5[dj, dk] = [(dj+2, dk+2, 1)]
    
    # M5 contributes to C11 (offset 0,0) with -1 and C12 (offset 0,2) with +1
    add_strassen_term(A_M5, B_M5, 0, 0, -1)  # C11
    add_strassen_term(A_M5, B_M5, 0, 2, 1)  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    A_M6 = np.empty((2, 2), dtype=object)
    B_M6 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M6[di, dj] = [(di+2, dj, 1), (di, dj, -1)]
    for dj in range(2):
        for dk in range(2):
            B_M6[dj, dk] = [(dj, dk, 1), (dj, dk+2, 1)]
    
    # M6 contributes to C22 (offset 2,2) with +1
    add_strassen_term(A_M6, B_M6, 2, 2, 1)  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    A_M7 = np.empty((2, 2), dtype=object)
    B_M7 = np.empty((2, 2), dtype=object)
    for di in range(2):
        for dj in range(2):
            A_M7[di, dj] = [(di, dj+2, 1), (di+2, dj+2, -1)]
    for dj in range(2):
        for dk in range(2):
            B_M7[dj, dk] = [(dj+2, dk, 1), (dj+2, dk+2, 1)]
    
    # M7 contributes to C11 (offset 0,0) with +1
    add_strassen_term(A_M7, B_M7, 0, 0, 1)  # C11
    
    # Part 2: Schoolbook for 4x4 x 4x3 (columns 4-6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k = 0,1,2 corresponds to B column 4,5,6
                U[i * m + j, col] = 1
                V[j * p + (k + 4), col] = 1
                W[(k + 4) * n + i, col] = 1
                col += 1
    
    # Verify col == rank
    # assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition with random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    val += A_test[i, j] * B_test[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i, k] += val
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col_fb] = 1
                    V_fb[j * p + k, col_fb] = 1
                    W_fb[k * n + i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
