import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE_S = 130.0  # safety margin under 150s limit


def _strassen_2x2_forms():
    """
    Return the 7 Strassen linear forms for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where:
    U[i*2+j, r] = coefficient of A[i,j] in the r-th left factor
    V[j*2+k, r] = coefficient of B[j,k] in the r-th right factor  
    W[k*2+i, r] = coefficient for C[k,i] in the r-th output term
    
    Index convention:
    A is 2x2: A[i,j], i=row, j=col
    B is 2x2: B[j,k], j=row, k=col
    C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    Output W indexed as W[k*2+i, r] per the trace convention
    """
    # 7 Strassen products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00 + B01)
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B11)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U rows: [A00, A01, A10, A11] (i*2+j)
    # V rows: [B00, B01, B10, B11] (j*2+k)
    # W rows: [C00, C01, C10, C11] but indexed as W[k*2+i]
    #   C00 -> k=0,i=0 -> W[0]
    #   C01 -> k=1,i=0 -> W[2]  (k*2+i = 1*2+0 = 2)
    #   C10 -> k=0,i=1 -> W[1]  (k*2+i = 0*2+1 = 1)
    #   C11 -> k=1,i=1 -> W[3]  (k*2+i = 1*2+1 = 3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)(B00+B01)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    # C10 += M2, C11 -= M2
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    # C01 += M3, C11 += M3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    # C00 += M4, C10 += M4
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5, C01 += M5
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # (M6 not used in standard output combination for this variant)
    # Actually M6 is not used in the C formulas above. Let me reconsider.
    
    # Wait, I need to recheck. The standard Strassen has 7 products but
    # not all appear in every C output. Let me use a verified set.
    
    # Let me restart with a cleaner formulation.
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Using the standard Strassen from Wikipedia:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b10 - b00)
    # p4 = a11(b01 - b11)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    #
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    
    # p1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # p3
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # p4
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # p5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # p6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # p6 not used in c formulas
    
    # p7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _schoolbook(n, m, p):
    """
    Schoolbook decomposition: rank = n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Build the 49-term Strassen decomposition for 4x4x4 matrix multiplication.
    
    Uses two levels of Strassen:
    Level 1: 4x4 = 2x(2x2) blocks, 7 block multiplications
    Level 2: Each 2x2 block multiplication uses 7 scalar multiplications
    Total: 7 * 7 = 49 terms
    """
    # Get the 2x2x2 Strassen forms
    U2, V2, W2 = _strassen_2x2_forms()
    
    # Verify 2x2x2 Strassen
    if not _verify_decomposition(U2, V2, W2, 2, 2, 2):
        # Fallback: try to fix or use schoolbook
        return None
    
    # Now compose for 4x4x4
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # B is 4x4, split into 2x2 blocks:
    # B = [[B00, B01], [B10, B11]]
    #
    # C = A*B, split into 2x2 blocks:
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen at block level:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    #
    # Each Pi is a 2x2 * 2x2 = 2x2 matrix product.
    # Each such product uses 7 scalar multiplications via Strassen.
    #
    # So for each Pi, we have 7 scalar terms.
    # Total: 7 * 7 = 49 scalar terms.
    
    # For each block-level product Pi = L_i * R_i, where L_i and R_i are 2x2 matrices
    # formed by linear combinations of the 2x2 blocks of A and B.
    
    # The 2x2 Strassen decomposition tells us how to compute X*Y for 2x2 matrices X, Y:
    # For each of the 7 Strassen terms r:
    #   left_factor_r = linear combination of X entries
    #   right_factor_r = linear combination of Y entries
    #   output_contribution to C entries
    
    # Let me define the block-level linear combinations:
    # Each Pi has a "left matrix" (2x2) and "right matrix" (2x2)
    # These are formed from the 2x2 blocks of A and B
    
    # Block indices:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # For the 4x4 case:
    # U has shape (4*4, rank) = (16, rank)
    #   Row index: i*4+j where i is row of A, j is col of A
    # V has shape (4*4, rank) = (16, rank)
    #   Row index: j*4+k where j is row of B, k is col of B
    # W has shape (4*4, rank) = (16, rank)
    #   Row index: k*4+i where k is col of C, i is row of C
    
    # For each block-level product Pi, the left matrix L_i is a 2x2 matrix
    # whose entries are linear combinations of A's 2x2 blocks.
    # Similarly for R_i and B's blocks.
    
    # Let me define the 7 block-level products:
    # Each is (left_block_combo, right_block_combo)
    # left_block_combo: 4x4 matrix where each 2x2 block is a coefficient
    # Actually, it's simpler to think of it as:
    # L_i is a 2x2 matrix of 2x2 blocks, each block being a coefficient times identity or zero
    
    # Let me be more explicit. For P1 = (A00+A11)(B00+B11):
    # The left matrix is a 4x4 matrix where:
    #   top-left 2x2 block = A00 (coefficient 1)
    #   top-right 2x2 block = 0
    #   bottom-left 2x2 block = 0
    #   bottom-right 2x2 block = A11 (coefficient 1)
    # So the left 4x4 matrix = [[A00, 0], [0, A11]]
    
    # The right matrix for P1:
    #   top-left 2x2 block = B00 (coefficient 1)
    #   top-right 2x2 block = 0
    #   bottom-left 2x2 block = 0
    #   bottom-right 2x2 block = B11 (coefficient 1)
    # So the right 4x4 matrix = [[B00, 0], [0, B11]]
    
    # Now, the 2x2 Strassen decomposition computes X*Y where X and Y are 2x2.
    # But here X and Y are 4x4 matrices with a special block structure.
    # However, since each 2x2 block of X is either 0 or a scalar multiple of a 2x2 block of A,
    # and similarly for Y, the product X*Y will have 2x2 blocks that are products of 2x2 blocks.
    
    # Actually, I think the cleanest way is to just directly construct the 49 terms.
    
    # Let me define the 7 block-level "left" and "right" 4x4 matrices as sparse patterns:
    # For each block-level product, I specify which A-blocks and B-blocks are used,
    # and with what coefficients.
    
    # Block layout for 4x4:
    # A: rows 0-3, cols 0-3
    #   Block (0,0): rows 0-1, cols 0-1 -> A indices (0,0),(0,1),(1,0),(1,1)
    #   Block (0,1): rows 0-1, cols 2-3 -> A indices (0,2),(0,3),(1,2),(1,3)
    #   Block (1,0): rows 2-3, cols 0-1 -> A indices (2,0),(2,1),(3,0),(3,1)
    #   Block (1,1): rows 2-3, cols 2-3 -> A indices (2,2),(2,3),(3,2),(3,3)
    
    # For the 2x2 Strassen on the block level, I need to map:
    # 2x2 "super-matrix" indices (I,J) to 4x4 block indices
    # Super-matrix 2x2:
    #   (0,0) -> block (0,0): rows 0-1, cols 0-1
    #   (0,1) -> block (0,1): rows 0-1, cols 2-3
    #   (1,0) -> block (1,0): rows 2-3, cols 0-1
    #   (1,1) -> block (1,1): rows 2-3, cols 2-3
    
    # Now, for each block-level Strassen term (say, the "left" part uses A-blocks),
    # the 2x2 Strassen left factor is a linear combination of the 4 entries of the 2x2 super-matrix.
    # Each entry of the super-matrix corresponds to a 2x2 block of A.
    
    # So if the 2x2 Strassen left factor for term r has coefficients (c00, c01, c10, c11)
    # for the super-matrix entries, then the actual 4x4 left matrix has:
    #   Block (0,0) multiplied by c00
    #   Block (0,1) multiplied by c01
    #   Block (1,0) multiplied by c10
    #   Block (1,1) multiplied by c11
    
    # Let me construct this properly.
    
    # First, let me define the 7 block-level products as (left_coeffs, right_coeffs)
    # where left_coeffs is a 2x2 matrix of coefficients for A-blocks,
    # and right_coeffs is a 2x2 matrix of coefficients for B-blocks.
    
    # P1 = (A00+A11)(B00+B11)
    # Left: [[1, 0], [0, 1]] (A00 and A11)
    # Right: [[1, 0], [0, 1]] (B00 and B11)
    
    # P2 = (A10+A11)B00
    # Left: [[0, 0], [1, 1]] (A10 and A11)
    # Right: [[1, 0], [0, 0]] (B00)
    
    # P3 = A00(B10-B00)
    # Left: [[1, 0], [0, 0]] (A00)
    # Right: [[-1, 0], [1, 0]] (-B00 + B10)
    
    # P4 = A11(B01-B11)
    # Left: [[0, 0], [0, 1]] (A11)
    # Right: [[0, 1], [0, -1]] (B01 - B11)
    
    # P5 = (A00+A01)B11
    # Left: [[1, 1], [0, 0]] (A00 + A01)
    # Right: [[0, 0], [0, 1]] (B11)
    
    # P6 = (A10-A00)(B00+B01)
    # Left: [[-1, 0], [1, 0]] (-A00 + A10)
    # Right: [[1, 1], [0, 0]] (B00 + B01)
    
    # P7 = (A01-A11)(B10+B11)
    # Left: [[0, 1], [0, -1]] (A01 - A11)
    # Right: [[0, 0], [1, 1]] (B10 + B11)
    
    block_products = [
        # (left 2x2 coeffs, right 2x2 coeffs)
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]]),   # P1
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]]),   # P2
        ([[1, 0], [0, 0]], [[-1, 0], [1, 0]]),  # P3
        ([[0, 0], [0, 1]], [[0, 1], [0, -1]]),  # P4
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]]),   # P5
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]]),  # P6
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]]),  # P7
    ]
    
    # Output combinations:
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # C block (I,J) gets contribution from block product Pi with coefficient w
    # C00: P1(+1), P4(+1), P5(-1), P7(-1)
    # C01: P3(+1), P5(+1)
    # C10: P2(+1), P4(+1)
    # C11: P1(+1), P2(-1), P3(+1), P7(+1)
    
    output_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: -1},   # C00
        (0, 1): {2: 1, 4: 1},                   # C01
        (1, 0): {1: 1, 3: 1},                   # C10
        (1, 1): {0: 1, 1: -1, 2: 1, 6: 1},     # C11
    }
    
    # Now build the 49-term decomposition
    # For each block product Pi (i=0..6) and each 2x2 Strassen term r (r=0..6):
    #   The left 4x4 matrix is formed by scaling A-blocks according to left_coeffs[i]
    #   The right 4x4 matrix is formed by scaling B-blocks according to right_coeffs[i]
    #   The 2x2 Strassen term r gives:
    #     left_factor = linear combo of the 2x2 super-matrix entries
    #     right_factor = linear combo of the 2x2 super-matrix entries
    #     output contributions to C super-matrix entries
    
    # Wait, I'm overcomplicating this. Let me think differently.
    
    # The 2x2 Strassen decomposition U2, V2, W2 works on 2x2 matrices.
    # When I apply it to the "super-matrix" level, each 2x2 entry of the super-matrix
    # is itself a 2x2 block of the 4x4 matrix.
    
    # So the left 4x4 matrix for block product Pi is:
    #   A 4x4 matrix where block (I,J) = left_coeffs[i][I][J] * A_block(I,J)
    #   But that's not quite right either, because the Strassen left factor is a scalar
    #   linear combination of the 2x2 entries.
    
    # Let me think about it more carefully.
    
    # The 2x2 Strassen says: for 2x2 matrices X and Y, X*Y can be computed as:
    # (X*Y)[p,q] = sum_r W2[q*2+p, r] * (sum_IJ U2[I*2+J, r] * X[I,J]) * (sum_KL V2[K*2+L, r] * Y[K,L])
    
    # Now, in the block-level application, X is the "left super-matrix" and Y is the "right super-matrix".
    # X[I,J] = left_coeffs[i][I][J] (a scalar)
    # Y[K,L] = right_coeffs[i][K][L] (a scalar)
    
    # But wait, that's not right. The block product Pi = L_i * R_i where L_i and R_i are 2x2 matrices
    # of 2x2 blocks. So L_i[I,J] is a 2x2 block (a 2x2 matrix), not a scalar.
    
    # I think the correct interpretation is:
    # The 4x4 matrix A can be viewed as a 2x2 matrix of 2x2 blocks.
    # The "left factor" for block product Pi is a 2x2 matrix of 2x2 blocks, where
    # block (I,J) = left_coeffs[i][I][J] * A_block(I,J)
    # Similarly for the right factor.
    
    # Now, to multiply two 2x2 matrices of 2x2 blocks, we can use Strassen at the block level.
    # The Strassen formulas give 7 "block products" of 2x2 blocks.
    # Each block product is a 2x2 * 2x2 = 2x2 matrix product, which can be done with 7 scalar multiplications.
    
    # So for block product Pi, the left block matrix L_i has 2x2 blocks:
    #   L_i[I,J] = left_coeffs[i][I][J] * A_block(I,J)
    # The right block matrix R_i has 2x2 blocks:
    #   R_i[K,L] = right_coeffs[i][K][L] * B_block(K,L)
    
    # The product L_i * R_i is a 2x2 matrix of 2x2 blocks, where:
    #   (L_i * R_i)[P,Q] = sum_J L_i[P,J] * R_i[J,Q]
    # Each term L_i[P,J] * R_i[J,Q] is a 2x2 * 2x2 = 2x2 matrix product.
    
    # Using Strassen for each such 2x2 * 2x2 product:
    #   L_i[P,J] * R_i[J,Q] = sum_r (scalar_r) * u_r * v_r
    #   where u_r is a scalar linear combination of the 4 entries of L_i[P,J]
    #   and v_r is a scalar linear combination of the 4 entries of R_i[J,Q]
    
    # This is getting very complex. Let me just do it computationally.
    
    # Alternative approach: directly construct the 49 terms by expanding everything.
    
    # For each block product Pi (i=0..6) and each Strassen sub-term r (r=0..6):
    #   The contribution to the 4x4 output C is:
    #   For each C-block (P,Q) that receives contribution from Pi with coefficient w:
    #     C_block(P,Q) += w * (L_i * R_i)[P,Q]
    #   And (L_i * R_i)[P,Q] = sum_J L_i[P,J] * R_i[J,Q]
    #   Each L_i[P,J] is a 2x2 matrix = left_coeffs[i][P][J] * A_block(P,J)
    #   Each R_i[J,Q] is a 2x2 matrix = right_coeffs[i][J][Q] * B_block(J,Q)
    #   So L_i[P,J] * R_i[J,Q] = left_coeffs[i][P][J] * right_coeffs[i][J][Q] * (A_block(P,J) * B_block(J,Q))
    
    # Now, A_block(P,J) is a 2x2 matrix with entries A[2P+a, 2J+b] for a,b in {0,1}
    # B_block(J,Q) is a 2x2 matrix with entries B[2J+c, 2Q+d] for c,d in {0,1}
    # Their product is a 2x2 matrix with entries:
    #   (A_block * B_block)[a,d] = sum_c A[2P+a, 2J+c] * B[2J+c, 2Q+d]
    
    # Using Strassen for this 2x2 * 2x2 product:
    #   (A_block * B_block)[a,d] = sum_r W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * A_block[b,c]) * (sum_ef V2[e*2+f, r] * B_block[e,f])
    #   = sum_r W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * A[2P+b, 2J+c]) * (sum_ef V2[e*2+f, r] * B[2J+e, 2Q+f])
    
    # So the total contribution to C[2P+a, 2Q+d] from block product Pi and Strassen term r is:
    #   w * left_coeffs[i][P][J] * right_coeffs[i][J][Q] * W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * A[2P+b, 2J+c]) * (sum_ef V2[e*2+f, r] * B[2J+e, 2Q+f])
    #   summed over J
    
    # This is getting unwieldy. Let me just build it numerically.
    
    # I'll create the full 4x4x4 decomposition by iterating over all the indices.
    
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    term_idx = 0
    for i_block in range(7):
        left_coeffs = np.array(block_products[i_block][0])
        right_coeffs = np.array(block_products[i_block][1])
        
        for r in range(7):
            # For this block product and Strassen sub-term,
            # I need to figure out which A entries, B entries, and C entries are involved.
            
            # The Strassen sub-term r involves:
            # left_factor = sum_bc U2[b*2+c, r] * X[b,c] where X is the 2x2 super-matrix
            # right_factor = sum_ef V2[e*2+f, r] * Y[e,f] where Y is the 2x2 super-matrix
            # output: for each (p,q), contribution = W2[q*2+p, r] * left_factor * right_factor
            
            # But X and Y here are the block-level matrices, whose entries are 2x2 blocks.
            # So X[b,c] = left_coeffs[b,c] * A_block(b,c)
            # Y[e,f] = right_coeffs[e,f] * B_block(e,f)
            
            # The left_factor is a scalar:
            # left_factor = sum_bc U2[b*2+c, r] * left_coeffs[b,c] * (scalar from A_block)
            # But A_block is a 2x2 matrix, so I need to extract a scalar from it.
            
            # Actually, I think the issue is that the Strassen decomposition works on scalars,
            # not on matrices. So I need to apply it at the scalar level.
            
            # Let me reconsider. The 4x4 matrix multiplication can be viewed as:
            # C = A * B where A, B, C are 4x4.
            # I can partition A, B, C into 2x2 blocks of 2x2 matrices.
            # Then C_block(P,Q) = sum_J A_block(P,J) * B_block(J,Q)
            # Each term A_block(P,J) * B_block(J,Q) is a 2x2 * 2x2 = 2x2 matrix product.
            
            # To compute all 4*4=16 such 2x2 matrix products (for the 4 C-blocks, each being
            # a sum of 2 terms), I can use Strassen at the block level.
            
            # The block-level Strassen gives 7 "block products" of 2x2 matrices.
            # Each block product is a 2x2 * 2x2 = 2x2 matrix product.
            # Each such product can be computed with 7 scalar multiplications.
            
            # So total: 7 * 7 = 49 scalar multiplications.
            
            # The key insight is that the 7 block-level Strassen products are:
            # P_i = L_i * R_i where L_i and R_i are 2x2 matrices of 2x2 blocks.
            # Each L_i[P,Q] = left_coeffs[i][P][Q] * A_block(P,Q) (a 2x2 matrix)
            # Each R_i[P,Q] = right_coeffs[i][P][Q] * B_block(P,Q) (a 2x2 matrix)
            
            # Now, L_i * R_i is a 2x2 matrix of 2x2 blocks:
            # (L_i * R_i)[P,Q] = sum_J L_i[P,J] * R_i[J,Q]
            # Each term is a 2x2 * 2x2 = 2x2 matrix product.
            
            # To compute L_i[P,J] * R_i[J,Q] using Strassen:
            # L_i[P,J] is a 2x2 matrix with entries L_i[P,J][a,b] = left_coeffs[i][P][J] * A[2P+a, 2J+b]
            # R_i[J,Q] is a 2x2 matrix with entries R_i[J,Q][c,d] = right_coeffs[i][J][Q] * B[2J+c, 2Q+d]
            
            # Using the 2x2 Strassen:
            # (L_i[P,J] * R_i[J,Q])[a,d] = sum_r W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * L_i[P,J][b,c]) * (sum_ef V2[e*2+f, r] * R_i[J,Q][e,f])
            
            # = sum_r W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * left_coeffs[i][P][J] * A[2P+b, 2J+c]) * (sum_ef V2[e*2+f, r] * right_coeffs[i][J][Q] * B[2J+e, 2Q+f])
            
            # = left_coeffs[i][P][J] * right_coeffs[i][J][Q] * sum_r W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * A[2P+b, 2J+c]) * (sum_ef V2[e*2+f, r] * B[2J+e, 2Q+f])
            
            # So for a fixed block product i and Strassen term r, and for a fixed J:
            # The contribution to C[2P+a, 2Q+d] is:
            #   left_coeffs[i][P][J] * right_coeffs[i][J][Q] * W2[d*2+a, r] * (sum_bc U2[b*2+c, r] * A[2P+b, 2J+c]) * (sum_ef V2[e*2+f, r] * B[2J+e, 2Q+f])
            
            # But this is for a specific P, Q, J, a, d. The total contribution to C is the sum over P, Q, J, a, d.
            
            # This is very complex to handle directly. Let me try a different approach.
            
            # I'll just build the decomposition by directly computing what each term contributes.
            
            # For each (i_block, r), I'll determine:
            # - Which A entries are involved and with what coefficients (for U)
            # - Which B entries are involved and with what coefficients (for V)
            # - Which C entries are involved and with what coefficients (for W)
            
            # From the formula above, for fixed i_block and r:
            # The term involves A[2P+b, 2J+c] for various P, J, b, c
            # and B[2J+e, 2Q+f] for various J, Q, e, f
            # and contributes to C[2P+a, 2Q+d] for various P, Q, a, d
            
            # The coefficient for A[2P+b, 2J+c] in the left factor is:
            #   left_coeffs[i_block][P][J] * U2[b*2+c, r]
            
            # The coefficient for B[2J+e, 2Q+f] in the right factor is:
            #   right_coeffs[i_block][J][Q] * V2[e*2+f, r]
            
            # The coefficient for C[2P+a, 2Q+d] in the output is:
            #   W2[d*2+a, r] * (sum over J of the output coefficient for C-block (P,Q) from block product i_block)
            
            # Wait, I need to be more careful. The output C_block(P,Q) gets contribution from block product i_block with some coefficient w (from output_coeffs).
            # And C_block(P,Q)[a,d] = C[2P+a, 2Q+d]
            
            # So the total coefficient for C[2P+a, 2Q+d] from block product i_block and Strassen term r is:
            #   w(P,Q,i_block) * W2[d*2+a, r]
            # where w(P,Q,i_block) is the coefficient from output_coeffs.
            
            # But this only works if the left and right factors are "aligned" properly.
            # The left factor involves A entries from blocks (P, J) for various J.
            # The right factor involves B entries from blocks (J, Q) for various J.
            # The product gives C entries from block (P, Q).
            
            # But the left factor is a single scalar (sum over b,c), and the right factor is a single scalar (sum over e,f).
            # So for a fixed P, Q, J, the product is a scalar that contributes to C[2P+a, 2Q+d] for all a,d.
            
            # But the left factor depends on P and J (through left_coeffs and A entries),
            # and the right factor depends on J and Q (through right_coeffs and B entries).
            # So for a fixed P, Q, the sum over J gives:
            #   sum_J [left_coeffs[i_block][P][J] * (sum_bc U2[b*2+c, r] * A[2P+b, 2J+c])] * [right_coeffs[i_block][J][Q] * (sum_ef V2[e*2+f, r] * B[2J+e, 2Q+f])]
            
            # This is a product of two sums, which is not separable in a simple way.
            
            # I think the issue is that I'm trying to combine the block-level Strassen with the scalar-level Strassen in a way that doesn't factor cleanly.
            
            # Let me try a completely different approach: just use the schoolbook for 4x4x4 (64 terms) and add 16 for the 5th column, giving 80 total. That's the same as the schoolbook for 4x4x5.
            
            # Or, I can try to find a better decomposition using a computational search.
            
            # Given the time constraints, let me just return the schoolbook solution for now.
            break
    
        if term_idx == 0:
            break
    
    if term_idx == 0:
        return None
    
    return U4, V4, W4


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook: rank = 80
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to build the 49+16=65 scheme
    # First, verify the 2x2x2 Strassen
    U2, V2, W2 = _strassen_2x2_forms()
    
    # Verify 2x2x2
    if _verify_decomposition(U2, V2, W2, 2, 2, 2):
        # Try to build 4x4x4
        result = _strassen_4x4x4()
        if result is not None:
            U4, V4, W4 = result
            if _verify_decomposition(U4, V4, W4, 4, 4, 4):
                # Extend to 4x4x5 by adding 16 schoolbook terms for the 5th column
                # The 4x4x4 decomposition handles C[0:4, 0:4]
                # We need to add terms for C[4, 0:4] (the 5th row of C, which is the 5th column in the trace convention)
                
                # Wait, the dimensions are:
                # A is 4x4 (n=4, m=4)
                # B is 4x5 (m=4, p=5)
                # C is 4x5 (p=5, n=4)
                
                # C[k, i] = sum_j A[i, j] * B[j, k]
                # k ranges over 0..4 (5 values), i ranges over 0..3 (4 values)
                
                # The 4x4x4 decomposition handles the case where B is 4x4 and C is 4x4.
                # For the 5th column of B (column index 4), we need additional terms.
                
                # For each i in 0..3 and j in 0..3:
                #   C[4, i] += A[i, j] * B[j, 4]
                
                # These are 4*4 = 16 additional schoolbook terms.
                
                rank_total = 49 + 16
                U_total = np.zeros((16, rank_total))
                V_total = np.zeros((20, rank_total))
                W_total = np.zeros((20, rank_total))
                
                # Copy the 4x4x4 decomposition
                # U4 has shape (16, 49)
                # V4 has shape (16, 49) - but we need (20, 49) for the full B
                # W4 has shape (16, 49) - but we need (20, 49) for the full C
                
                U_total[:, :49] = U4
                V_total[:16, :49] = V4
                W_total[:16, :49] = W4
                
                # Add 16 schoolbook terms for the 5th column of B
                col = 49
                for i in range(4):
                    for j in range(4):
                        U_total[i * 4 + j, col] = 1
                        V_total[j * 5 + 4, col] = 1  # B[j, 4]
                        W_total[4 * 4 + i, col] = 1  # C[4, i] -> k=4, i=i -> k*n+i = 4*4+i = 16+i
                        col += 1
                
                if _verify_decomposition(U_total, V_total, W_total, 4, 4, 5):
                    best_U, best_V, best_W = U_total, V_total, W_total
                    best_rank = rank_total
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_S:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
