"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition with rank 22:
    # - For j in {0,1}: use Strassen-like rank-7 decomposition for 2x2 blocks (A_{i,j} B_{j,k})
    # - For j = 2: use rank-8 outer product (A_{i,2} B_{2,k})
    #
    # Matrix multiplication tensor: T[i,j,k] = A[i,j] * B[j,k]
    # Output convention: W[k*n + i, r]
    #
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (8, rank)
    #
    # U index: i*m + j (row of A)
    # V index: j*p + k (column of B)
    # W index: k*n + i (output C[i,k])
    #
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # For each j in {0,1}, apply Strassen to 2x2 block (i in {0,1}, k in {0,1} or {2,3})
    for j in range(2):
        # Strassen for C = A * B where A is 2x2, B is 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        #
        # M0 = a00 * (b00 + b11)
        # M1 = (a00 + a01) * b10
        # M2 = (a10 + a11) * b00
        # M3 = a11 * (b10 - b00)
        # M4 = (a00 + a11) * (b00 + b11)
        # M5 = (a01 - a11) * (b10 + b11)
        # M6 = (a00 - a10) * (b01 + b11)
        #
        # c00 = M0 + M3 - M4 + M6
        # c01 = M1 + M4
        # c10 = M2 + M5
        # c11 = M0 - M1 + M2 + M5

        # j=0: k in {0,1}; j=1: k in {2,3}
        k0 = j * 2  # 0 for j=0, 2 for j=1
        k1 = k0 + 1

        # M0 = A_{0,j} * (B_{j, k0} + B_{j, k1})
        U[0 * m + j][col] = 1          # A_{0,j}
        V[j * p + k0][col] = 1         # B_{j, k0}
        V[j * p + k1][col] = 1         # B_{j, k1}
        W[k0 * n + 0][col] = 1         # C_{0, k0}
        W[k1 * n + 0][col] = 1         # C_{0, k1}
        W[k0 * n + 1][col] = -1        # C_{1, k0} (from M0 + M3 - M4 + M6)
        W[k1 * n + 1][col] = 1         # C_{1, k1} (from M0 - M1 + M2 + M5)
        col += 1

        # M1 = (A_{0,j} + A_{1,j}) * B_{j, k1}
        U[0 * m + j][col] = 1          # A_{0,j}
        U[1 * m + j][col] = 1          # A_{1,j}
        V[j * p + k1][col] = 1         # B_{j, k1}
        W[k1 * n + 0][col] = 1         # C_{0, k1}
        W[k1 * n + 1][col] = -1        # C_{1, k1} (from M0 - M1 + M2 + M5)
        col += 1

        # M2 = (A_{1,j} + A_{1,j+1?}) -- wait, Strassen uses 2x2 blocks
        # Actually for j=0, A is 2x2 with rows i=0,1 and cols j=0,1? No.
        # The 2x2 block in A is fixed at column j, rows i=0,1. But Strassen needs 2x2 A.
        #
        # Wait, I think I'm confusing myself. Let me reconsider.
        #
        # For each fixed j, A_{i,j} is a column of A (2x1), and B_{j,k} is a row of B (1x4).
        # So for fixed j, we're computing a 2x4 output C_i,k = A_i,j * B_j,k.
        #
        # This is NOT a 2x2 matrix multiplication. It's a 2x1 times 1x4, which is outer product.
        # Strassen doesn't apply directly.
        #
        # I need to rethink the decomposition.

    # Let me restart with a cleaner approach.
    #
    # The tensor T has entries T[i, j, k] for i in 0..1, j in 0..2, k in 0..3.
    # T[i, j, k] = A[i, j] * B[j, k]
    #
    # We want to decompose T as sum over r of U[i,j, r] * V[j, k, r] * W[k, i, r]
    # where U has shape (6, rank), V has shape (12, rank), W has shape (8, rank).
    #
    # One simple decomposition: for each (i, j, k) triple, use one rank-1 term.
    # This gives rank 2*3*4 = 24.
    #
    # To reduce rank, we can group terms. For fixed j, the 2x4 block C_i,k = A_i,j * B_j,k
    # has rank 2 (as a 2x4 matrix). So for each j, we need at least 2 terms to represent
    # the 2x4 outer product.
    #
    # Actually, A_i,j * B_j,k for fixed j is a rank-1 matrix (outer product of a 2-vector and 4-vector).
    # So it needs only 1 term.
    #
    # So for each j, 1 term suffices. Total rank = 3.
    #
    # But wait, the tensor is T[i, j, k], not a matrix. The decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For fixed j, T[i, j, k] = A[i, j] * B[j, k] = (column j of A) * (row j of B)
    #
    # This can be written as: U[i*m+j, r] = A[i, j], V[j*p+k, r] = B[j, k], W[k*n+i, r] = 1
    # But U and V can't depend on the input values A and B; they must be constants.
    #
    # So we need constant matrices U, V, W such that the identity holds for all A, B.
    #
    # The schoolbook decomposition: for each (i, j, k), one term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # Rank = 24.
    #
    # To reduce rank, we need to find linear dependencies.
    #
    # Let me try a different approach: group by j.
    # For fixed j, we have 2*4 = 8 terms (i in 0..1, k in 0..3).
    # These are: A[i, j] * B[j, k] for i=0,1 and k=0,1,2,3.
    #
    # This is the outer product of vector (A[0,j], A[1,j]) and vector (B[j,0], B[j,1], B[j,2], B[j,3]).
    #
    # In the tensor decomposition, this corresponds to 8 basis tensors.
    #
    # Can we represent this outer product with fewer than 8 terms?
    #
    # The rank of the 2x4 matrix (for fixed j) is 1, but in the tensor sense, we're decomposing
    # a 3-tensor, not a matrix.
    #
    # Actually, for fixed j, the sub-tensor T[i, j, k] = A[i,j] * B[j,k] is separable:
    # T[i, j, k] = sum over (u, v) of alpha_u * A[i,j] * beta_v * B[j,k]
    #
    # But alpha_u * A[i,j] * beta_v * B[j,k] can be absorbed into U, V, W.
    #
    # For fixed j, T[i, j, k] = A[i, j] * B[j, k]
    #
    # This can be written as a single term: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # But this would require U, V, W to depend on i and k simultaneously, which they do via their indices.
    #
    # Wait, I think I'm overcomplicating this.
    #
    # Let me just use the schoolbook decomposition with rank 24 for now, and see if I can find
    # a better one.
    #
    # Actually, let me think about this more carefully.
    #
    # The tensor T is of size 2 x 3 x 4 = 24 entries.
    # Each entry T[i, j, k] = A[i, j] * B[j, k].
    #
    # A rank-r decomposition is: T = sum_{r=1}^r U_r \otimes V_r \otimes W_r
    # where U_r is a 2x3 matrix, V_r is a 3x4 matrix, W_r is a 4x2 matrix.
    #
    # In the given format:
    # U[i*m+j, r] = U_r[i, j]
    # V[j*p+k, r] = V_r[j, k]
    # W[k*n+i, r] = W_r[k, i]
    #
    # The condition is: for all i, j, k:
    # sum_r U_r[i, j] * V_r[j, k] * W_r[k, i] = A[i, j] * B[j, k]
    #
    # This must hold for all A, B. So the coefficients must match.
    #
    # Let's write this as a linear system. For each (i, j, k), we have:
    # sum_r U_r[i, j] * V_r[j, k] * W_r[k, i] = delta_{A[i,j]} * delta_{B[j,k]}
    #
    # This is not a standard linear system because of the product U_r[i,j] * V_r[j,k] * W_r[k,i].
    #
    # Let me try small rank values and see if I can find a solution.
    #
    # Actually, let me just output the schoolbook rank-24 decomposition for now.

    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
