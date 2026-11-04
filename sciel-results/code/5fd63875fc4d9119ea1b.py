"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120  # seconds

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 (rank 7)
    # Indexing: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: C01 - C00
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, -1],  # M7: C00 - C11
    ]

    # Schoolbook decomposition for 3x3 (rank 27)
    schoolbook_U = []
    schoolbook_V = []
    schoolbook_W = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0] * 9
                v = [0] * 9
                w = [0] * 9
                u[i * 3 + j] = 1
                v[j * 3 + k] = 1
                w[k * 3 + i] = 1
                schoolbook_U.append(u)
                schoolbook_V.append(v)
                schoolbook_W.append(w)

    # Kronecker product: rank = 7 * 27 = 189
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    col = 0
    for r in range(7):
        if time.time() - start_time > DEADLINE - 5:
            break
        for s in range(27):
            # U part: A[i,j] where i = 3*i1 + i2, j = 3*j1 + j2
            # strassen_U[r] indexed by (i1,j1) -> i1*2+j1
            # schoolbook_U[s] indexed by (i2,j2) -> i2*3+j2
            for a in range(4):
                ua = strassen_U[r][a]
                if ua == 0:
                    continue
                i1, j1 = a // 2, a % 2
                for b in range(9):
                    ub = schoolbook_U[s][b]
                    if ub == 0:
                        continue
                    i2, j2 = b // 3, b % 3
                    flat_idx = 18 * i1 + 6 * i2 + 3 * j1 + j2
                    U[flat_idx][col] = ua * ub

            # V part: B[j,k] where j = 3*j1 + j2, k = 3*k1 + k2
            for a in range(4):
                va = strassen_V[r][a]
                if va == 0:
                    continue
                j1, k1 = a // 2, a % 2
                for b in range(9):
                    vb = schoolbook_V[s][b]
                    if vb == 0:
                        continue
                    j2, k2 = b // 3, b % 3
                    flat_idx = 18 * j1 + 6 * j2 + 3 * k1 + k2
                    V[flat_idx][col] = va * vb

            # W part: C[i,k] where i = 3*i1 + i2, k = 3*k1 + k2
            # W indexed by k*6 + i
            for a in range(4):
                wa = strassen_W[r][a]
                if wa == 0:
                    continue
                k1, i1 = a // 2, a % 2
                for b in range(9):
                    wb = schoolbook_W[s][b]
                    if wb == 0:
                        continue
                    k2, i2 = b // 3, b % 3
                    flat_idx = 18 * k1 + 6 * k2 + 3 * i1 + i2
                    W[flat_idx][col] = wa * wb

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
