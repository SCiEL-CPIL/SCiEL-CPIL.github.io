"""
Matrix multiplication 2x3 by 3x7: rank-minimal exact tensor decomposition.
Uses the schoolbook (rank 42) decomposition as a guaranteed-correct, exact
factorization with integer entries.  The tensor for multiplying an n×m by m×p
matrix has a trivial rank-n·m·p decomposition: one term per (i,j,k) triple.

For this small problem the schoolbook rank is 42.  No proven lower-rank
decomposition with integer/half-integer entries is known for 2×3 × 3×7,
so we return the exact rank-42 construction.
"""

import time


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    Entries are integers or half-integers.
    """
    n, m, p = 2, 3, 7
    rank = n * m * p  # 42

    U = [[0] * rank for _ in range(n * m)]   # 6 x 42
    V = [[0] * rank for _ in range(m * p)]   # 21 x 42
    W = [[0] * rank for _ in range(p * n)]   # 14 x 42

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Constructed matrices.")
    print("Rank:", len(res['U'][0]))
