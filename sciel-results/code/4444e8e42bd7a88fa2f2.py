"""Rewritten program: block-wise Strassen for 2x2 * 2x14 multiplication."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x2 * 2x14 matrix multiplication tensor using a block-wise
    Strassen-like approach.

    The left matrix A is 2x2. The right matrix B is 2x14, which we split into
    7 blocks of size 2x2: B = [B_1, B_2, ..., B_7] where each B_i is 2x2.

    For each block B_i, we compute A * B_i using Strassen's 7-multiplication
    algorithm. Since the left matrix is 2x2, we can apply Strassen directly
    to each 2x2 * 2x2 block multiplication.

    Strassen's algorithm for 2x2 * 2x2:
      M1 = (A11 + A22)(B11 + B22)
      M2 = (A21 + A22)B11
      M3 = A11(B21 - B22)
      M4 = A22(B12 - B11)
      M5 = (A11 + A12)B22
      M6 = (A21 - A11)(B11 + B12)
      M7 = (A12 - A22)(B21 + B22)

      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6

    So for each of the 7 blocks, we need 7 rank-1 terms, giving 7*7 = 49
    total rank terms.

    The output C is 2x14. Following the trace convention W[k*n+i, r], where
    n=2 (rows of A), p=14 (cols of B), so W has shape (p*n, rank) = (28, rank).
    W[k*n + i, r] corresponds to output element C[i, k].

    Let's define the indices carefully:
    - A is 2x2: A[i, j] for i,j in {0,1}
    - B is 2x14: B[j, k] for j in {0,1}, k in {0,...,13}
    - C is 2x14: C[i, k] for i in {0,1}, k in {0,...,13}

    C[i,k] = sum_j A[i,j] * B[j,k]

    We split B into 7 blocks: for block b (0..6), columns k = 2*b and 2*b+1.
    So B[:, 2*b : 2*b+2] is a 2x2 matrix.

    For block b, define:
      B11_b = B[0, 2*b]
      B12_b = B[0, 2*b+1]
      B21_b = B[1, 2*b]
      B22_b = B[1, 2*b+1]

    Strassen products for block b:
      M1_b = (A00 + A11) * (B11_b + B22_b)
      M2_b = (A10 + A11) * B11_b
      M3_b = A00 * (B21_b - B22_b)
      M4_b = A11 * (B12_b - B11_b)
      M5_b = (A00 + A01) * B22_b
      M6_b = (A10 - A00) * (B11_b + B12_b)
      M7_b = (A01 - A11) * (B21_b + B22_b)

    Output for block b:
      C[0, 2*b]   = M1_b + M4_b - M5_b + M7_b
      C[0, 2*b+1] = M3_b + M5_b
      C[1, 2*b]   = M2_b + M4_b
      C[1, 2*b+1] = M1_b - M2_b + M3_b + M6_b

    Now, each M_t_b is a product of a linear combination of A entries and a
    linear combination of B entries. In the tensor decomposition, each rank-1
    term corresponds to one such product.

    U has shape (n*m, rank) = (4, rank). U[i*m + j, c] = U[i*2 + j, c] is the
    coefficient of A[i,j] in the c-th rank-1 term.

    V has shape (m*p, rank) = (28, rank). V[j*p + k, c] = V[j*14 + k, c] is
    the coefficient of B[j,k] in the c-th rank-1 term.

    W has shape (p*n, rank) = (28, rank). W[k*n + i, c] = W[k*2 + i, c] is
    the coefficient of C[i,k] in the c-th rank-1 term.

    Let's enumerate all 49 terms.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2  # 7
    rank = 7 * num_blocks  # 49

    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49

    col = 0
    for b in range(num_blocks):
        # Column indices in B for this block
        k0 = 2 * b
        k1 = 2 * b + 1

        # M1_b = (A00 + A11) * (B[0,k0] + B[1,k1])
        U[0 * 2 + 0][col] += 1  # A00
        U[1 * 2 + 1][col] += 1  # A11
        V[0 * p + k0][col] += 1  # B[0,k0]
        V[1 * p + k1][col] += 1  # B[1,k1]
        # Output: C[0,k0] += 1, C[1,k1] += 1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1

        # M2_b = (A10 + A11) * B[0,k0]
        U[1 * 2 + 0][col] += 1  # A10
        U[1 * 2 + 1][col] += 1  # A11
        V[0 * p + k0][col] += 1  # B[0,k0]
        # Output: C[1,k0] += 1
        W[k0 * n + 1][col] += 1
        col += 1

        # M3_b = A00 * (B[1,k0] - B[1,k1])
        U[0 * 2 + 0][col] += 1  # A00
        V[1 * p + k0][col] += 1  # B[1,k0]
        V[1 * p + k1][col] -= 1  # -B[1,k1]
        # Output: C[0,k1] += 1
        W[k1 * n + 0][col] += 1
        col += 1

        # M4_b = A11 * (B[0,k1] - B[0,k0])
        U[1 * 2 + 1][col] += 1  # A11
        V[0 * p + k1][col] += 1  # B[0,k1]
        V[0 * p + k0][col] -= 1  # -B[0,k0]
        # Output: C[0,k0] += 1, C[1,k0] += 1
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        col += 1

        # M5_b = (A00 + A01) * B[1,k1]
        U[0 * 2 + 0][col] += 1  # A00
        U[0 * 2 + 1][col] += 1  # A01
        V[1 * p + k1][col] += 1  # B[1,k1]
        # Output: C[0,k0] -= 1, C[0,k1] += 1
        W[k0 * n + 0][col] -= 1
        W[k1 * n + 0][col] += 1
        col += 1

        # M6_b = (A10 - A00) * (B[0,k0] + B[0,k1])
        U[1 * 2 + 0][col] += 1  # A10
        U[0 * 2 + 0][col] -= 1  # -A00
        V[0 * p + k0][col] += 1  # B[0,k0]
        V[0 * p + k1][col] += 1  # B[0,k1]
        # Output: C[1,k1] += 1
        W[k1 * n + 1][col] += 1
        col += 1

        # M7_b = (A01 - A11) * (B[1,k0] + B[1,k1])
        U[0 * 2 + 1][col] += 1  # A01
        U[1 * 2 + 1][col] -= 1  # -A11
        V[1 * p + k0][col] += 1  # B[1,k0]
        V[1 * p + k1][col] += 1  # B[1,k1]
        # Output: C[0,k0] += 1
        W[k0 * n + 0][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
