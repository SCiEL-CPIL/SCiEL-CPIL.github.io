"""Matrix multiplication tensor decomposition for 3x4 x 4x8."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start = time.time()
    deadline = start + 145.0

    n, m, p = DIMENSIONS
    rank_school = n * m * p

    # Strategy: Split k into two blocks of 4.
    # For each k-block, decompose 3x4x4:
    #   - Row 0: 1x4x4, schoolbook rank 16
    #   - Rows 1-2: 2x4x4, Strassen on two 2x2x2 blocks, rank 14
    #   Total per k-block: 30
    # Total rank: 60

    if time.time() > deadline - 5:
        # Fallback to schoolbook
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen factors for 2x2x2
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11 + C22
        [0, 0, 1, 0],   # M2: C21
        [0, 0, 0, -1],  # M3: -C22
        [-1, 0, 1, 0],  # M4: -C11 + C21
        [0, 1, 0, 1],   # M5: C12 + C22
        [1, 0, 0, 0],   # M6: C11
        [0, 1, 0, 0]    # M7: C12
    ]

    for k_block in range(2):
        k_start = k_block * 4

        # Part 1: Row 0, 1x4x4 schoolbook (rank 16)
        for j in range(4):
            for k_local in range(4):
                k = k_start + k_local
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1

        # Part 2: Rows 1-2, 2x4x4 using Strassen (rank 14)
        # Split j into 2 blocks of 2: [0,1], [2,3]
        for j_block in range(2):
            j_start = j_block * 2
            for t in range(7):
                # U factors: map local (i', j') to global (i, j)
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += strassen_U[t][l]

                # V factors: map local (j', k') to global (j, k)
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += strassen_V[t][l]

                # W factors: map local (i', k') to global (i, k)
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i][col] += strassen_W[t][l]

                col += 1

    if col != rank:
        # Fallback to schoolbook
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
