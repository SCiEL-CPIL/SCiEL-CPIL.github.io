import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 12
    U = [[0] * rank for _ in range(n * m)]   # 8 x 12
    V = [[0] * rank for _ in range(m * p)]   # 24 x 12
    W = [[0] * rank for _ in range(p * n)]   # 12 x 12

    # Per-output-column rank-2 decomposition.
    # For output column k (0..5):
    #   C[0,k] = sum_j A[0,j] * B[j,k]
    #   C[1,k] = sum_j A[1,j] * B[j,k]
    # Split the 4 inner indices into two groups of 2: {0,1} and {2,3}.
    # Group g (g=0 -> indices {0,1}, g=1 -> indices {2,3}) contributes one
    # rank-1 term to this column.
    for k in range(p):
        for g in range(2):
            c = k * 2 + g   # global rank index
            for i in range(n):            # rows of A / rows of C
                for j in range(2):         # local index within group
                    jj = 2 * g + j        # actual inner index
                    U[i * m + jj][c] += 1  # A row i, inner jj
            for j in range(2):
                jj = 2 * g + j
                V[jj * p + k][c] += 1      # B inner jj, col k
            for i in range(n):
                W[k * n + i][c] += 1       # C col k, row i (trace convention)

    return {"U": U, "V": V, "W": W}
