import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 5, 6

    # Strategy: block-diagonal Kronecker decomposition
    # Split m=5 into 1 + 4
    # Part 1: 2x1x6, rank = 12
    # Part 2: 2x4x6, split k=6 into 2+2+2, each 2x4x2
    #   Each 2x4x2: split j=4 into 2+2, each 2x2x2 has rank 7 (Strassen)
    #   So 2x4x2 has rank 14
    #   2x4x6 has rank 3*14 = 42
    # Total rank = 12 + 42 = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: j=0, A[:,0] (2x1) * B[0,:] (1x6) -> 2x6
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # Strassen terms for 2x2x2:
    strassen_terms = [
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        ),
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        ),
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        ),
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]

    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = 1 + j_block * 2

            for term in strassen_terms:
                u_entries, v_entries, w_entries = term

                for (li, lj, coeff) in u_entries:
                    gi = li
                    gj = j_base + lj
                    U[gi * m + gj][col] += coeff

                for (lj, lk, coeff) in v_entries:
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col] += coeff

                for (li, lk, coeff) in w_entries:
                    gi = li
                    gk = k_base + lk
                    W[gk * n + gi][col] += coeff

                col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verification step to ensure correctness
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return a rank-60 trivial decomposition
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
