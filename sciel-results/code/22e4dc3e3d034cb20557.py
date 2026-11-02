"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6

    # Rank-15 decomposition using Kronecker structure.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We use the fact that for fixed j, the sub-tensor is a Kronecker product
    # of a 2-dim identity-like tensor and a 6-dim identity-like tensor.
    # Actually, let's use a direct rank-15 construction:
    #
    # For each of the 3 columns of A (and 3 rows of B), we need to compute
    # 3 outer products of size 2x6. But we can batch them.
    #
    # Key insight: The tensor T has the form sum_j (A_col_j ⊗ B_row_j) transposed appropriately.
    # A_col_j is 2x1, B_row_j is 1x6. Their outer product is 2x6.
    # The tensor for one such outer product (with indices i, k) has rank 6
    # (since C[i,k] = A_i * B_k, which is a rank-1 matrix in the (i,k) plane
    # but the tensor indices are (i,j), (j,k), (k,i)).
    #
    # Let me just build a rank-15 decomposition directly:
    # Use 5 terms per j (3 values of j), total 15.
    # For each j, we split the 2x6 computation into 5 rank-1 tensors.
    #
    # Actually, the simplest valid improvement is to use the known fact that
    # matrix multiplication M2x3 * M3x6 can be done in rank 15 using
    # the following construction:
    #
    # Let me use the standard Kronecker decomposition:
    # T[i,j, j,k, k,i] = sum_{r=0}^{4} U_r[i,j] V_r[j,k] W_r[k,i]
    # where r indexes 5 terms per j... this doesn't quite work that way.
    #
    # Let me just verify a rank-18 construction first (better than 36):
    # Split into 3 independent 2x1 * 1x6 multiplications, each of rank 6.
    # Total rank = 3 * 6 = 18.

    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18

    col = 0
    for j in range(m):  # 3 columns
        for k in range(p):  # 6 rows
            # This term corresponds to A[i,j] * B[j,k] contributing to C[i,k]
            # U row index: i*m + j
            # V row index: j*p + k
            # W row index: k*n + i
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1

    # Wait, that's 3*6=18 terms, but each term sets 2 entries in U, 1 in V, 2 in W.
    # Let me re-check: for fixed j,k, we have n=2 terms (one for each i).
    # So total columns = 3 * 6 * 2 = 36. That's the schoolbook!

    # Let me try a different approach. For fixed j,k, the contribution to C
    # is A[:,j] * B[j,k] (a column vector scaled). The tensor entries are:
    # T[i*m+j, j*p+k, k*n+i] = A[i,j]*B[j,k] for all i.
    # This is a rank-2 tensor in the i-index: sum_i u_i[i] v_i[i] w_i[i]
    # where u_i[i*m+j]=1, v_i[j*p+k]=1, w_i[k*n+i]=1.
    # So each (j,k) pair needs rank n=2.
    # Total rank = m * p * n = 3*6*2 = 36. Back to schoolbook.

    # Let me try to find a genuine improvement using Strassen-like ideas.
    # For 2x2 matrix multiplication, Strassen uses 7 multiplications instead of 8.
    # For 2x3 * 3x6, we can pad to 3x3 * 3x6 or use other techniques.

    # Actually, let me try a rank-27 construction:
    # Pad A to 3x3 (adding a zero row), pad B to 3x6.
    # Then use a 3x3 * 3x6 multiplication.
    # 3x3 * 3x6 = (3x3)*(3x2*3) -> 3x2 blocks of 3x3*3x1? No.
    # 3x3 * 3x6: split into three 3x3 * 3x2 multiplications.
    # 3x3 * 3x2: can we do better than 18?
    # This is getting complicated. Let me just return the schoolbook for now
    # but structured cleanly.

    rank = n * m * p
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
