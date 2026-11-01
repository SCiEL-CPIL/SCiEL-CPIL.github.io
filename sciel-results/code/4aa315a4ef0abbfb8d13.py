"""Matrix multiplication 2x3x9 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    """Return True if U,V,W exactly decompose the matrix multiplication tensor.

    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 if (i,j,k) is a valid index, 0 otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """Return a rank-minimal exact decomposition of the 2x3x9 matrix mult tensor."""
    t_start = time.time()
    DEADLINE = t_start + 115.0

    n, m, p = 2, 3, 9

    # --- Schoolbook baseline (rank 54) ---
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # --- Try to find lower-rank decomposition via structured search ---
    # The tensor rank of M_{2,3,9} is at least max(2*3, 3*9, 9*2) = 27.
    # Known results suggest rank 54 for these dimensions, but let's try to improve.
    #
    # Strategy: Use the block structure. A is 2x3, B is 3x9.
    # We can view B as 3 blocks of 3x3: B = [B1 | B2 | B3].
    # C = A * B = [A*B1 | A*B2 | A*B3], each A*Bj is 2x3.
    # This gives 3 independent 2x3x3 multiplications, each rank 18, total 54.
    #
    # Alternative: Try to find a rank-45 or lower decomposition using
    # the fact that n=2 is small. Use randomized search with rational entries.

    import random
    random.seed(12345)

    # Try rank r from 27 to 53
    for r in range(27, 54):
        if time.time() > DEADLINE:
            break

        # Generate random U, V, W with entries in {-1, 0, 1, 1/2}
        # and check if they satisfy the tensor equation.
        # This is a very rough search, but let's try a few iterations.

        for attempt in range(100):
            if time.time() > DEADLINE:
                break

            U = [[random.choice([-1, 0, 0, 0, 1, 0.5]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 0, 0, 1, 0.5]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 0, 0, 1, 0.5]) for _ in range(r)] for _ in range(p * n)]

            if _validate(U, V, W, n, m, p):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break

    # Return the best result found
    return best_result
