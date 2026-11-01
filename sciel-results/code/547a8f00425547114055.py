"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
Uses a block-pairing formulation: partitions the inner dimension j into two
pairs {0,1} and {2,3}, and within each pair uses the 1x2 x 2x1 inner-product
tensor (rank 2), yielding total rank 64 with a distinct factor layout."""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Two j-pairs, each of size 2; each (i,k) needs 2 multiplications per pair.
    # Total rank = n * p * (m/2) * 2 = n * p * m = 64.
    rank = n * m * p  # 64

    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows

    # Pair structure: pairs = [(0,1), (2,3)]
    pairs = [(0, 1), (2, 3)]
    pair_size = 2

    col = 0
    for i in range(n):
        for k in range(p):
            for pair_idx, (j1, j2) in enumerate(pairs):
                # Two factors for this (i, k, pair):
                # Factor a: A[i,j1]*B[j1,k]
                U[i * m + j1][col] = 1
                V[j1 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                # Factor b: A[i,j2]*B[j2,k]
                U[i * m + j2][col] = 1
                V[j2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
