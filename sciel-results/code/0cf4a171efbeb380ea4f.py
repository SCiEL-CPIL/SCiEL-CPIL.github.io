"""Matrix multiplication 4x4x8 with a rank-98 decomposition."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """
    Build the rank-7 Strassen decomposition for 2x2x2.
    Returns U (4,7), V (4,7), W (4,7) where:
      U[i*m+j, r] = coefficient for A[i,j] in term r
      V[j*p+k, r] = coefficient for B[j,k] in term r
      W[k*n+i, r] = coefficient for C[i,k] in term r
    """
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11

    # M3: A00(B10-B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11

    # M4: A11(B01-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = -1 # C00
    W[2, 3] = 1  # C10

    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C00
    W[1, 4] = -1 # C01

    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11

    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C00
    W[1, 6] = 1  # C01

    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    A is 4x4, B is 4x4, C is 4x4.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _build_2x2x2_strassen()
    # U2: (4,7) for 2x2 A, V2: (4,7) for 2x2 B, W2: (4,7) for 2x2 C

    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)

    # For each of the 7 Strassen terms, we need to compute the 4x4 block
    # combinations. The 4x4 matrix A is split into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # Similarly B and C.
    #
    # For each Strassen term t (0..6):
    #   The A-combination is a 4x4 matrix formed from the 2x2 block coefficients
    #   The B-combination is a 4x4 matrix
    #   The C-combination is a 4x4 matrix
    #
    # Each of these 4x4 matrices is then decomposed using the 2x2x2 Strassen
    # decomposition, giving 7 sub-terms.
    #
    # Total: 7 * 7 = 49 terms.

    # For each Strassen term, determine which 2x2 blocks of A, B, C are used
    # and with what coefficients.
    #
    # Strassen term 0 (M1): A=(A00+A11), B=(B00+B11), C contributes to C00 and C11
    # Strassen term 1 (M2): A=(A10+A11), B=B00, C contributes to C10 and C11
    # Strassen term 2 (M3): A=A00, B=(B10-B00), C contributes to C01 and C11
    # Strassen term 3 (M4): A=A11, B=(B01-B11), C contributes to C00 and C10
    # Strassen term 4 (M5): A=(A00+A01), B=B11, C contributes to C00 and C01
    # Strassen term 5 (M6): A=(A10-A00), B=(B00+B01), C contributes to C11
    # Strassen term 6 (M7): A=(A01-A11), B=(B10+B11), C contributes to C00 and C01

    # For each term, the A-block is a 4x4 matrix. We need to express it as
    # a linear combination of the 2x2 block basis.
    #
    # Let's define the block structure:
    # A 4x4 matrix has 16 entries. We index them as:
    # A[i,j] where i,j in 0..3
    # Block (bi,bj) where bi,bj in 0..1 contains A[2*bi:2*bi+2, 2*bj:2*bj+2]
    #
    # For the Strassen term, the A-combination is a 4x4 matrix where:
    # - Some 2x2 blocks are set to specific 2x2 matrices (like all-ones or identity)
    # - Other blocks are zero
    #
    # Actually, the Strassen decomposition works on 2x2 blocks. So for a 4x4
    # multiplication, we treat the 4x4 matrices as 2x2 block matrices where
    # each block is 2x2. Then we apply Strassen to get 7 block products,
    # each of which is a 2x2 * 2x2 = 2x2 matrix multiplication.
    #
    # So the structure is:
    # For each of the 7 Strassen terms at the block level:
    #   - The A-block combination gives a 4x4 matrix (2x2 blocks of 2x2)
    #   - The B-block combination gives a 4x4 matrix
    #   - The C-block combination gives a 4x4 matrix
    #
    # Each 2x2 * 2x2 product is decomposed using the 2x2x2 Strassen (rank 7).
    #
    # So for Strassen term t at the block level, and sub-term s at the 2x2 level:
    #   col = t * 7 + s
    #   U[:, col] = (A-block for term t) * U2[:, s]  (reshaped)
    #   V[:, col] = (B-block for term t) * V2[:, s]  (reshaped)
    #   W[:, col] = (C-block for term t) * W2[:, s]  (reshaped)
    #
    # Wait, that's not quite right. Let me think more carefully.
    #
    # The 4x4x4 tensor is:
    # T[i,j,k] = sum over A[i,j] * B[j,k] * C[i,k]
    #
    # We want to decompose this as sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    #
    # Using Strassen at the block level:
    # Let A be split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # The 7 Strassen products at the block level are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # And:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2 matrix (block product). Each block product is a
    # 2x2 * 2x2 multiplication, which has rank 7 (Strassen).
    #
    # So for each Mj, we have 7 scalar terms. Total: 7*7 = 49 terms.
    #
    # For term (j, s) where j is the Strassen block term and s is the 2x2 sub-term:
    #   The contribution to the 4x4x4 tensor is:
    #   U[i,j,r] * V[j,k,r] * W[i,k,r]
    #
    #   Where:
    #   - U[i,j,r] comes from the A-block combination for Mj, evaluated at position (i,j)
    #   - V[j,k,r] comes from the B-block combination for Mj, evaluated at position (j,k)
    #   - W[i,k,r] comes from the C-block combination for Mj, evaluated at position (i,k)
    #
    #   And the 2x2 sub-term s gives the specific scalar coefficients.

    # Let's define the block-level Strassen coefficients more carefully.
    # For each block-level term j (0..6), we have:
    #   A_j: a 2x2 matrix of coefficients for the 2x2 blocks of A
    #   B_j: a 2x2 matrix of coefficients for the 2x2 blocks of B
    #   C_j: a 2x2 matrix of coefficients for the 2x2 blocks of C
    #
    # Where A_j[bi, bj] means the coefficient of block A[bi,bj] in term j.
    #
    # Actually, it's simpler. For each block-level term j:
    #   The A-part is a specific 4x4 matrix (with 2x2 block structure)
    #   The B-part is a specific 4x4 matrix
    #   The C-part is a specific 4x4 matrix
    #
    # And the 2x2 sub-decomposition gives the scalar coefficients.

    # Let me just build it directly.
    #
    # For block-level term j, the A-combination is:
    # j=0: A00 + A11 -> blocks (0,0) and (1,1) have identity, others zero
    # j=1: A10 + A11 -> blocks (1,0) and (1,1) have identity
    # j=2: A00 -> block (0,0) has identity
    # j=3: A11 -> block (1,1) has identity
    # j=4: A00 + A01 -> blocks (0,0) and (0,1) have identity
    # j=5: A10 - A00 -> block (1,0) has identity, block (0,0) has -identity
    # j=6: A01 - A11 -> block (0,1) has identity, block (1,1) has -identity
    #
    # Similarly for B and C.

    # For each block-level term j, define which 2x2 blocks are non-zero and their signs.
    # A_blocks[j] = list of (block_row, block_col, sign)
    # B_blocks[j] = list of (block_row, block_col, sign)
    # C_blocks[j] = list of (block_row, block_col, sign)

    A_blocks = [
        [(0, 0, 1), (1, 1, 1)],  # j=0: A00 + A11
        [(1, 0, 1), (1, 1, 1)],  # j=1: A10 + A11
        [(0, 0, 1)],             # j=2: A00
        [(1, 1, 1)],             # j=3: A11
        [(0, 0, 1), (0, 1, 1)],  # j=4: A00 + A01
        [(1, 0, 1), (0, 0, -1)], # j=5: A10 - A00
        [(0, 1, 1), (1, 1, -1)], # j=6: A01 - A11
    ]

    B_blocks = [
        [(0, 0, 1), (1, 1, 1)],  # j=0: B00 + B11
        [(0, 0, 1)],             # j=1: B00
        [(1, 0, 1), (0, 0, -1)], # j=2: B10 - B00
        [(0, 1, 1), (1, 1, -1)], # j=3: B01 - B11
        [(1, 1, 1)],             # j=4: B11
        [(0, 0, 1), (0, 1, 1)],  # j=5: B00 + B01
        [(1, 0, 1), (1, 1, 1)],  # j=6: B10 + B11
    ]

    C_blocks = [
        [(0, 0, 1), (1, 1, 1)],  # j=0: C00 + C11
        [(1, 0, 1), (1, 1, -1)], # j=1: C10 - C11
        [(0, 1, 1), (1, 1, 1)],  # j=2: C01 + C11
        [(0, 0, -1), (1, 0, 1)], # j=3: -C00 + C10
        [(0, 0, 1), (0, 1, -1)], # j=4: C00 - C01
        [(1, 1, 1)],             # j=5: C11
        [(0, 0, -1), (0, 1, 1)], # j=6: -C00 + C01
    ]

    # Now build the full 4x4x4 decomposition.
    for j in range(7):
        for s in range(7):
            col = j * 7 + s
            # U[:, col]: for each (i, j) in 4x4, the coefficient
            # This comes from the A-blocks for term j, combined with U2[:, s]
            for (br, bc, sign) in A_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        i = 2 * br + li
                        jj = 2 * bc + lj
                        U[i * 4 + jj, col] += sign * U2[li * 2 + lj, s]

            # V[:, col]: for each (j, k) in 4x4
            for (br, bc, sign) in B_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        jj = 2 * br + li
                        k = 2 * bc + lj
                        V[jj * 4 + k, col] += sign * V2[li * 2 + lj, s]

            # W[:, col]: for each (i, k) in 4x4
            for (br, bc, sign) in C_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        i = 2 * br + li
                        k = 2 * bc + lj
                        W[k * 4 + i, col] += sign * W2[li * 2 + lj, s]

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _build_4x4x8_rank98():
    """
    Build the rank-98 decomposition for 4x4x8 by splitting B into two 4x4 halves.
    C = A * B where A is 4x4, B is 4x8.
    Split B = [B1 | B2] where B1, B2 are 4x4.
    C = [A*B1 | A*B2] = [C1 | C2]
    Each A*Bk is a 4x4x4 multiplication with rank 49.
    Total rank = 49 + 49 = 98.
    """
    U44, V44, W44 = _build_4x4x4_strassen()

    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)

    # First half: C1 = A * B1
    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44

    # Second half: C2 = A * B2
    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    n, m, p = 4, 4, 8

    # Build the rank-98 Strassen-based decomposition
    U, V, W = _build_4x4x8_rank98()

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if verification fails
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
