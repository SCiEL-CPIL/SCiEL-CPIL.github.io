import time
import sys

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using Strassen's algorithm at the block level (2x2 blocks of 2x2 matrices).
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 block coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Define linear forms for X (from A) and Y (from B) for each block term s (0..6)
    # X is 2x2, Y is 2x2.
    # A blocks: A00, A01, A10, A11. Indices (I,J) -> A[2I:2I+2, 2J:2J+2]
    # B blocks: B00, B01, B10, B11. Indices (J,K) -> B[2J:2J+2, 2K:2K+2]
    
    # For each block term s, X is a linear combo of A blocks, Y is a linear combo of B blocks.
    # We represent X and Y as 2x2 matrices of coefficients for the 4 A-blocks and 4 B-blocks.
    # Actually, it's easier to define the linear form on the 4 scalar elements of the 2x2 block.
    # But the blocks themselves are 2x2 matrices of scalars.
    # Let's define the mapping from global indices to local block indices.
    
    # Helper to add coefficients to U, V, W for a specific scalar multiplication term.
    # term_id: 0..48
    # s = term_id // 7 (block term 0..6)
    # u = term_id % 7  (scalar Strassen term 0..6)
    
    # Strassen 2x2 scalar terms S0..S6 for matrices X, Y:
    # S0 = (x00+x11)(y00+y11)
    # S1 = (x10+x11)y00
    # S2 = x00(y01-y11)
    # S3 = x11(y10-y00)
    # S4 = (x00+x01)y11
    # S5 = (x10-x00)(y00+y01)
    # S6 = (x01-x11)(y10+y11)
    
    # Coefficients for X (linear form on X's elements x00, x01, x10, x11)
    # Each S_u is Lx(X) * Ly(Y).
    # We need Lx and Ly for each u.
    
    # Lx coefficients for each u (index 0..6)
    # x00, x01, x10, x11
    Lx_coeffs = [
        [1, 0, 0, 1], # S0: x00 + x11
        [0, 0, 1, 1], # S1: x10 + x11
        [1, 0, 0, 0], # S2: x00
        [0, 0, 0, 1], # S3: x11
        [1, 1, 0, 0], # S4: x00 + x01
        [0, 0, 1, -1],# S5: x10 - x00
        [0, 1, 0, -1] # S6: x01 - x11
    ]
    
    # Ly coefficients for each u
    # y00, y01, y10, y11
    Ly_coeffs = [
        [1, 0, 0, 1], # S0: y00 + y11
        [1, 0, 0, 0], # S1: y00
        [0, 1, 0, -1],# S2: y01 - y11
        [0, 0, 1, -1],# S3: y10 - y00
        [0, 0, 0, 1], # S4: y11
        [1, 1, 0, 0], # S5: y00 + y01
        [0, 0, 1, 1]  # S6: y10 + y11
    ]
    
    # Coefficients for M_s elements (m00, m01, m10, m11) from S_u
    # m00 = S0 + S3 - S4 + S6
    # m01 = S2 + S4
    # m10 = S1 + S3
    # m11 = S0 - S1 + S2 + S5
    M_coeffs = [
        [1, 0, 0, 1], # S0 -> m00:1, m11:1
        [0, 0, 1, -1],# S1 -> m10:1, m11:-1
        [0, 1, 0, 1], # S2 -> m01:1, m11:1
        [1, 0, 1, 0], # S3 -> m00:1, m10:1
        [-1, 1, 0, 0],# S4 -> m00:-1, m01:1
        [0, 0, 0, 1], # S5 -> m11:1
        [1, 0, 0, 0]  # S6 -> m00:1
    ]
    # Note: M_coeffs[u] gives coefficients for [m00, m01, m10, m11]
    
    # Block combination coefficients for C blocks from M_s
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    C_coeffs = [
        [1, 0, 0, 1], # M0 -> C00:1, C11:1
        [0, 0, 1, -1],# M1 -> C10:1, C11:-1
        [0, 1, 0, 1], # M2 -> C01:1, C11:1
        [1, 0, 1, 0], # M3 -> C00:1, C10:1
        [-1, 1, 0, 0],# M4 -> C00:-1, C01:1
        [0, 0, 0, 1], # M5 -> C11:1
        [1, 0, 0, 0]  # M6 -> C00:1
    ]
    # Note: C_coeffs[s] gives coefficients for [C00, C01, C10, C11]
    
    # Define linear combinations of A-blocks for X in each block term s
    # X is 2x2. X[I,J] is a linear combo of A-blocks.
    # A-blocks are 2x2 matrices.
    # We need to map local X indices (I,J) to global A indices.
    # X[I,J] = sum_{I',J'} coeff_A[s][I,J][I',J'] * A[I',J']
    
    # Let's define coeff_A[s] as a dict or 4D array: coeff_A[s][I][J][I'][J']
    # Similarly for B.
    
    # s=0: M0 = (A00+A11)(B00+B11)
    # X00 = A00, X01 = 0, X10 = 0, X11 = A11? No.
    # Strassen block algorithm:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Let's define X and Y for each s.
    # X is 2x2, Y is 2x2.
    # s=0: X = [[A00, 0], [0, A11]]? No, the product is (A00+A11)(B00+B11).
    # This is not a standard 2x2 matrix product of two 2x2 matrices formed by A and B blocks.
    # Strassen's algorithm for 4x4 using 2x2 blocks:
    # We treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    # The multiplication is performed using 7 multiplications of 2x2 block matrices.
    # Each "multiplication" is a product of two 2x2 block matrices.
    # Let's define the 7 block-matrix products.
    
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Each P_s is a product of two 2x2 matrices (which are themselves 2x2 blocks of scalars? No.)
    # Wait. A00 is a 2x2 matrix of scalars.
    # So P1 is a product of two 2x2 matrices of scalars?
    # No, A00+A11 is a 2x2 matrix of scalars.
    # So P1 is a 2x2 matrix of scalars resulting from multiplying two 2x2 matrices of scalars.
    # This requires 7 scalar multiplications (Strassen for 2x2).
    # So yes, my previous logic holds.
    
    # We need to define the 2x2 matrices X_s and Y_s for each s.
    # X_s and Y_s are 2x2 matrices where each entry is a linear combination of A-blocks or B-blocks.
    # But A-blocks are 2x2 matrices of scalars.
    # So X_s is a 2x2 matrix of 2x2 matrices?
    # No. The "2x2 matrix" in the Strassen step is the block matrix.
    # Let's clarify:
    # We are multiplying A (4x4) and B (4x4).
    # We split A into 4 blocks A_ij (2x2).
    # We split B into 4 blocks B_ij (2x2).
    # We compute 7 products P_s = X_s * Y_s, where X_s and Y_s are 2x2 matrices formed from the blocks.
    # For example, for P1, X1 = [[A00, 0], [0, A11]]? No.
    # P1 = (A00 + A11) * (B00 + B11).
    # This is a product of two 2x2 matrices (A00+A11) and (B00+B11).
    # So X1 is the 2x2 matrix (A00+A11). Y1 is the 2x2 matrix (B00+B11).
    # Wait, (A00+A11) is a single 2x2 matrix.
    # So P1 is the product of two 2x2 matrices.
    # This product is computed using Strassen's 7 scalar multiplications.
    
    # So for each s, we define X_s (a 2x2 matrix of scalars? No, a 2x2 matrix where entries are linear combos of A-block entries)
    # and Y_s (a 2x2 matrix where entries are linear combos of B-block entries).
    
    # Let's define X_s entries as linear combinations of A-block entries.
    # A-block entries are A[i,j].
    # X_s is 2x2. Let X_s[p,q].
    # X_s[p,q] = sum_{I,J} coeff_A[s][p,q][I,J] * A[I,J]
    # Here I,J are block indices (0,1). A[I,J] is a 2x2 matrix.
    # This is getting complicated.
    
    # Simpler:
    # For each s, X_s is a 2x2 matrix. Each element of X_s is a linear combination of the 16 elements of A.
    # Similarly for Y_s.
    # We can precompute these coefficients.
    
    # Let's define the coefficients for X_s and Y_s.
    # X_s is 2x2. Let's index its elements by (p,q) in {0,1}^2.
    # X_s[p,q] = sum_{i,j} c_{i,j} A[i,j].
    
    # s=0: P0 = (A00+A11)(B00+B11)
    # X0 = A00 + A11. This is a 2x2 matrix.
    # X0[0,0] = A[0,0] + A[2,2]
    # X0[0,1] = A[0,1] + A[2,3]
    # X0[1,0] = A[1,0] + A[3,2]
    # X0[1,1] = A[1,1] + A[3,3]
    # Y0 = B00 + B11.
    # Y0[0,0] = B[0,0] + B[2,2]
    # Y0[0,1] = B[0,1] + B[2,3]
    # Y0[1,0] = B[1,0] + B[3,2]
    # Y0[1,1] = B[1,1] + B[3,3]
    
    # s=1: P1 = (A10+A11)B00
    # X1 = A10 + A11
    # X1[0,0] = A[2,0] + A[2,2]
    # X1[0,1] = A[2,1] + A[2,3]
    # X1[1,0] = A[3,0] + A[3,2]
    # X1[1,1] = A[3,1] + A[3,3]
    # Y1 = B00
    # Y1[0,0] = B[0,0]
    # Y1[0,1] = B[0,1]
    # Y1[1,0] = B[1,0]
    # Y1[1,1] = B[1,1]
    
    # s=2: P2 = A00(B01-B11)
    # X2 = A00
    # Y2 = B01 - B11
    
    # s=3: P3 = A11(B10-B00)
    # X3 = A11
    # Y3 = B10 - B00
    
    # s=4: P4 = (A00+A01)B11
    # X4 = A00 + A01
    # Y4 = B11
    
    # s=5: P5 = (A10-A00)(B00+B01)
    # X5 = A10 - A00
    # Y5 = B00 + B01
    
    # s=6: P6 = (A01-A11)(B10+B11)
    # X6 = A01 - A11
    # Y6 = B10 + B11
    
    # We can define a helper to build the coefficient matrix for U and V.
    
    def get_block_coeffs(block_type, sign=1):
        # block_type: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns a 4x4 matrix of coefficients for A[i,j]
        # A00: rows 0,1 cols 0,1
        # A01: rows 0,1 cols 2,3
        # A10: rows 2,3 cols 0,1
        # A11: rows 2,3 cols 2,3
        coeffs = [[0]*4 for _ in range(4)]
        r_start = (block_type // 2) * 2
        c_start = (block_type % 2) * 2
        for r in range(2):
            for c in range(2):
                coeffs[r_start+r][c_start+c] = sign
        return coeffs

    # Define X_s and Y_s as 2x2 matrices of 4x4 coefficient matrices.
    # X_s[p][q] is a 4x4 matrix of coefficients for A.
    
    X_coeffs = []
    Y_coeffs = []
    
    # s=0
    X0 = [
        [get_block_coeffs(0) , get_block_coeffs(0)], # This is wrong. X0 is A00+A11.
        # A00+A11 is a 2x2 matrix.
        # Element (0,0) of X0 is A00[0,0] + A11[0,0].
        # So X0[0][0] = coeff(A00) + coeff(A11).
    ]
    
    # Let's build X_coeffs[s] as a 2x2 list of 4x4 matrices.
    
    def add_block_to_matrix(mat, block_idx, sign):
        # mat is 4x4
        bc = get_block_coeffs(block_idx, sign)
        for i in range(4):
            for j in range(4):
                mat[i][j] += bc[i][j]

    def init_4x4():
        return [[0]*4 for _ in range(4)]

    # s=0: X0 = A00+A11, Y0 = B00+B11
    X0 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    # X0 is a 2x2 matrix.
    # X0[0,0] corresponds to the top-left element of the 2x2 matrix (A00+A11).
    # Wait, A00+A11 is a 2x2 matrix.
    # So X0 IS the matrix A00+A11.
    # So X0[0,0] = A[0,0]+A[2,2], X0[0,1] = A[0,1]+A[2,3], etc.
    # So X0 is just one 2x2 matrix of coefficients?
    # No, X0 is a 2x2 matrix where each entry is a scalar linear combination.
    # So X0_coeffs should be a 2x2 matrix of 4x4 matrices?
    # No. X0 is a 2x2 matrix.
    # The elements of X0 are linear combinations of A's elements.
    # So X0_coeffs is a 2x2 matrix, where each element is a 4x4 matrix of coefficients.
    
    # Let's define X_coeffs[s] as a 2x2 matrix of 4x4 matrices.
    
    X_coeffs = []
    Y_coeffs = []
    
    # s=0
    X0 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X0[0][0], 0, 1) # A00
    add_block_to_matrix(X0[0][0], 3, 1) # A11
    add_block_to_matrix(X0[0][1], 0, 1) # A00
    add_block_to_matrix(X0[0][1], 3, 1) # A11
    add_block_to_matrix(X0[1][0], 0, 1) # A00
    add_block_to_matrix(X0[1][0], 3, 1) # A11
    add_block_to_matrix(X0[1][1], 0, 1) # A00
    add_block_to_matrix(X0[1][1], 3, 1) # A11
    X_coeffs.append(X0)
    
    Y0 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y0[0][0], 0, 1) # B00
    add_block_to_matrix(Y0[0][0], 3, 1) # B11
    add_block_to_matrix(Y0[0][1], 0, 1) # B00
    add_block_to_matrix(Y0[0][1], 3, 1) # B11
    add_block_to_matrix(Y0[1][0], 0, 1) # B00
    add_block_to_matrix(Y0[1][0], 3, 1) # B11
    add_block_to_matrix(Y0[1][1], 0, 1) # B00
    add_block_to_matrix(Y0[1][1], 3, 1) # B11
    Y_coeffs.append(Y0)
    
    # s=1: X1 = A10+A11, Y1 = B00
    X1 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X1[0][0], 2, 1) # A10
    add_block_to_matrix(X1[0][0], 3, 1) # A11
    add_block_to_matrix(X1[0][1], 2, 1) # A10
    add_block_to_matrix(X1[0][1], 3, 1) # A11
    add_block_to_matrix(X1[1][0], 2, 1) # A10
    add_block_to_matrix(X1[1][0], 3, 1) # A11
    add_block_to_matrix(X1[1][1], 2, 1) # A10
    add_block_to_matrix(X1[1][1], 3, 1) # A11
    X_coeffs.append(X1)
    
    Y1 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y1[0][0], 0, 1) # B00
    add_block_to_matrix(Y1[0][1], 0, 1) # B00
    add_block_to_matrix(Y1[1][0], 0, 1) # B00
    add_block_to_matrix(Y1[1][1], 0, 1) # B00
    Y_coeffs.append(Y1)
    
    # s=2: X2 = A00, Y2 = B01-B11
    X2 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X2[0][0], 0, 1)
    add_block_to_matrix(X2[0][1], 0, 1)
    add_block_to_matrix(X2[1][0], 0, 1)
    add_block_to_matrix(X2[1][1], 0, 1)
    X_coeffs.append(X2)
    
    Y2 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y2[0][0], 1, 1) # B01
    add_block_to_matrix(Y2[0][0], 3, -1) # B11
    add_block_to_matrix(Y2[0][1], 1, 1) # B01
    add_block_to_matrix(Y2[0][1], 3, -1) # B11
    add_block_to_matrix(Y2[1][0], 1, 1) # B01
    add_block_to_matrix(Y2[1][0], 3, -1) # B11
    add_block_to_matrix(Y2[1][1], 1, 1) # B01
    add_block_to_matrix(Y2[1][1], 3, -1) # B11
    Y_coeffs.append(Y2)
    
    # s=3: X3 = A11, Y3 = B10-B00
    X3 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X3[0][0], 3, 1)
    add_block_to_matrix(X3[0][1], 3, 1)
    add_block_to_matrix(X3[1][0], 3, 1)
    add_block_to_matrix(X3[1][1], 3, 1)
    X_coeffs.append(X3)
    
    Y3 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y3[0][0], 2, 1) # B10
    add_block_to_matrix(Y3[0][0], 0, -1) # B00
    add_block_to_matrix(Y3[0][1], 2, 1) # B10
    add_block_to_matrix(Y3[0][1], 0, -1) # B00
    add_block_to_matrix(Y3[1][0], 2, 1) # B10
    add_block_to_matrix(Y3[1][0], 0, -1) # B00
    add_block_to_matrix(Y3[1][1], 2, 1) # B10
    add_block_to_matrix(Y3[1][1], 0, -1) # B00
    Y_coeffs.append(Y3)
    
    # s=4: X4 = A00+A01, Y4 = B11
    X4 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X4[0][0], 0, 1)
    add_block_to_matrix(X4[0][0], 1, 1)
    add_block_to_matrix(X4[0][1], 0, 1)
    add_block_to_matrix(X4[0][1], 1, 1)
    add_block_to_matrix(X4[1][0], 0, 1)
    add_block_to_matrix(X4[1][0], 1, 1)
    add_block_to_matrix(X4[1][1], 0, 1)
    add_block_to_matrix(X4[1][1], 1, 1)
    X_coeffs.append(X4)
    
    Y4 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y4[0][0], 3, 1)
    add_block_to_matrix(Y4[0][1], 3, 1)
    add_block_to_matrix(Y4[1][0], 3, 1)
    add_block_to_matrix(Y4[1][1], 3, 1)
    Y_coeffs.append(Y4)
    
    # s=5: X5 = A10-A00, Y5 = B00+B01
    X5 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X5[0][0], 2, 1)
    add_block_to_matrix(X5[0][0], 0, -1)
    add_block_to_matrix(X5[0][1], 2, 1)
    add_block_to_matrix(X5[0][1], 0, -1)
    add_block_to_matrix(X5[1][0], 2, 1)
    add_block_to_matrix(X5[1][0], 0, -1)
    add_block_to_matrix(X5[1][1], 2, 1)
    add_block_to_matrix(X5[1][1], 0, -1)
    X_coeffs.append(X5)
    
    Y5 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y5[0][0], 0, 1)
    add_block_to_matrix(Y5[0][0], 1, 1)
    add_block_to_matrix(Y5[0][1], 0, 1)
    add_block_to_matrix(Y5[0][1], 1, 1)
    add_block_to_matrix(Y5[1][0], 0, 1)
    add_block_to_matrix(Y5[1][0], 1, 1)
    add_block_to_matrix(Y5[1][1], 0, 1)
    add_block_to_matrix(Y5[1][1], 1, 1)
    Y_coeffs.append(Y5)
    
    # s=6: X6 = A01-A11, Y6 = B10+B11
    X6 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(X6[0][0], 1, 1)
    add_block_to_matrix(X6[0][0], 3, -1)
    add_block_to_matrix(X6[0][1], 1, 1)
    add_block_to_matrix(X6[0][1], 3, -1)
    add_block_to_matrix(X6[1][0], 1, 1)
    add_block_to_matrix(X6[1][0], 3, -1)
    add_block_to_matrix(X6[1][1], 1, 1)
    add_block_to_matrix(X6[1][1], 3, -1)
    X_coeffs.append(X6)
    
    Y6 = [[init_4x4() for _ in range(2)] for _ in range(2)]
    add_block_to_matrix(Y6[0][0], 2, 1)
    add_block_to_matrix(Y6[0][0], 3, 1)
    add_block_to_matrix(Y6[0][1], 2, 1)
    add_block_to_matrix(Y6[0][1], 3, 1)
    add_block_to_matrix(Y6[1][0], 2, 1)
    add_block_to_matrix(Y6[1][0], 3, 1)
    add_block_to_matrix(Y6[1][1], 2, 1)
    add_block_to_matrix(Y6[1][1], 3, 1)
    Y_coeffs.append(Y6)
    
    # Now construct U, V, W
    for t in range(rank):
        s = t // 7
        u = t % 7
        
        # U coefficients
        # U[i*4+j, t] = sum_{p,q} Lx_coeffs[u][p*2+q] * X_coeffs[s][p][q][i][j]
        # Lx_coeffs[u] is [x00, x01, x10, x11]
        # X_coeffs[s][p][q] is 4x4 matrix for X[p,q]
        
        for i in range(4):
            for j in range(4):
                val = 0
                for p in range(2):
                    for q in range(2):
                        val += Lx_coeffs[u][p*2+q] * X_coeffs[s][p][q][i][j]
                U[i*4+j][t] = val
                
        # V coefficients
        # V[j*4+k, t] = sum_{p,q} Ly_coeffs[u][p*2+q] * Y_coeffs[s][p][q][j][k]
        # Note: Y is B. B[j,k].
        # Y_coeffs[s][p][q] is 4x4 matrix for Y[p,q] which is B[j,k].
        # So index is j*4+k.
        
        for j in range(4):
            for k in range(4):
                val = 0
                for p in range(2):
                    for q in range(2):
                        val += Ly_coeffs[u][p*2+q] * Y_coeffs[s][p][q][j][k]
                V[j*4+k][t] = val
                
        # W coefficients
        # W[k*4+i, t]
        # Contribution to C[i,k] from S_u in P_s.
        # S_u contributes to M_s elements m_{pq} with coeff M_coeffs[u][pq_idx]
        # M_s contributes to C_{IK} with coeff C_coeffs[s][IK_idx]
        # C_{IK} contains C[i,k] where i in block I, k in block K.
        # i = 2I + p, k = 2K + q.
        
        for I in range(2):
            for K in range(2):
                # C_coeffs[s] is [C00, C01, C10, C11]
                c_coeff = C_coeffs[s][I*2+K]
                if c_coeff == 0:
                    continue
                for p in range(2):
                    for q in range(2):
                        # M_coeffs[u] is [m00, m01, m10, m11]
                        m_coeff = M_coeffs[u][p*2+q]
                        if m_coeff == 0:
                            continue
                        i = 2*I + p
                        k = 2*K + q
                        W[k*4+i][t] += c_coeff * m_coeff
                        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
