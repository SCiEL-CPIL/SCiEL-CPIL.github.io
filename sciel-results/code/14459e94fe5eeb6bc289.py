"""Matrix multiplication 2x4 x 4x4: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use the standard schoolbook decomposition.
    # For multiplying an (n x m) matrix A by an (m x p) matrix B:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor indices: a = i*m + j  (size n*m = 8)
    #                 b = j*p + k  (size m*p = 16)
    #                 c = k*n + i  (size p*n = 8)  [official trace convention W[k*n+i, r]]
    #
    # The matrix multiplication tensor T[a,b,c] = 1 iff there exist i,j,k such that
    #   a = i*m + j, b = j*p + k, c = k*n + i
    #
    # Schoolbook: one rank-1 term per (i,j,k) triple.
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = n * m * p      # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
