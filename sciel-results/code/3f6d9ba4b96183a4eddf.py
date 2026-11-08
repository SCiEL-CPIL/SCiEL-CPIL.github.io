"""Matrix multiplication 3x4x8 rank-88 decomposition (Strassen-based)."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Rank 88:
    # i=0: 1x4x8 schoolbook -> 32
    # i=1,2: 2x4x8 using Strassen on 2x2 blocks -> 56
    # Total 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88

    col = 0

    # Part 1: i=0, schoolbook (1x4x8)
    for j in range(4):
        for k in range(8):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1

    # Part 2: i=1,2 using Strassen (2x4x8)
    # Strassen 2x2x2 factors
    sU = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
    sV = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    sW = [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]

    for jb in range(2): # j blocks of 2
        js = jb * 2
        for kb in range(4): # k blocks of 2
            ks = kb * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                # U
                for l in range(4):
                    val = sU[t][l]
                    if val:
                        li = l // 2
                        lj = l % 2
                        U[(1 + li) * m + js + lj][col] += val
                # V
                for l in range(4):
                    val = sV[t][l]
                    if val:
                        lj = l // 2
                        lk = l % 2
                        V[(js + lj) * p + ks + lk][col] += val
                # W
                for l in range(4):
                    val = sW[t][l]
                    if val:
                        li = l // 2
                        lk = l % 2
                        W[(ks + lk) * n + 1 + li][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
