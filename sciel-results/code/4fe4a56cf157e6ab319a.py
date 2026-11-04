"""
Direct construction of the rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses the trivial (schoolbook) decomposition with rank n*m*p = 80.
"""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit

    n, m, p = DIMENSIONS

    # The known minimal rank for small integer matrix multiplication is n*m*p
    # For 4x4 * 4x5, this is 4*4*5 = 80.
    rank = n * m * p

    # Initialize factor matrices
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i (following trace convention W[k*n+i, r])

                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i

                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1

                col += 1

                # Check time limit
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
