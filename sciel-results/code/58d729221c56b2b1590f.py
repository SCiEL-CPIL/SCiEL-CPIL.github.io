import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Standard Strassen forms (using 1-indexed S1..S7, stored as t=0..6)
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # S1: (a00+a11)(b00+b11)
    # S2: (a10+a11)b00
    # S3: a11(b10-b00)
    # S4: (a00+a01)b11
    # S5: (a00-a10)(b00+b10)
    # S6: (a11-a01)(b01+b11)
    # S7: (a01-a11)(b10+b11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S5 + S6
    # C10 = S2 + S3
    # C11 = S1 - S2 + S3 + S7

    U_forms = [
        [1, 0, 0, 1],    # S1: a00 + a11
        [0, 0, 1, 1],    # S2: a10 + a11
        [0, 0, 0, 1],    # S3: a11
        [1, 1, 0, 0],    # S4: a00 + a01
        [1, 0, -1, 0],   # S5: a00 - a10
        [0, -1, 0, 1],   # S6: a11 - a01
        [0, 1, 0, -1]    # S7: a01 - a11
    ]

    num_blocks = p // 2
    for u in range(num_blocks):
        for t in range(7):
            c = u * 7 + t
            if time.time() > deadline:
                break

            for i in range(4):
                U[i][c] = U_forms[t][i]

            v00 = 0 * p + 2 * u
            v01 = 0 * p + 2 * u + 1
            v10 = 1 * p + 2 * u
            v11 = 1 * p + 2 * u + 1

            w00 = (2 * u) * n + 0
            w01 = (2 * u + 1) * n + 0
            w10 = (2 * u) * n + 1
            w11 = (2 * u + 1) * n + 1

            if t == 0:  # S1
                V[v00][c] += 1
                V[v11][c] += 1
                W[w00][c] += 1
                W[w11][c] += 1
            elif t == 1:  # S2
                V[v00][c] += 1
                W[w10][c] += 1
            elif t == 2:  # S3
                V[v10][c] += 1
                V[v00][c] -= 1
                W[w10][c] += 1
                W[w11][c] += 1
            elif t == 3:  # S4
                V[v11][c] += 1
                W[w00][c] += 1
            elif t == 4:  # S5
                V[v00][c] += 1
                V[v10][c] += 1
                W[w00][c] -= 1
                W[w01][c] += 1
            elif t == 5:  # S6
                V[v01][c] += 1
                V[v11][c] += 1
                W[w01][c] += 1
            elif t == 6:  # S7
                V[v10][c] += 1
                V[v11][c] += 1
                W[w00][c] += 1
                W[w11][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
