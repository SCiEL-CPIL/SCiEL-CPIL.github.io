"""Matrix multiplication 3x4x8 using rank-88 tensor decomposition."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 118  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    rank = 88

    # U: (n*m, rank) = (12, 88)
    # V: (m*p, rank) = (32, 88)
    # W: (p*n, rank) = (24, 88)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: row i=0, schoolbook 1x4x8 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Part 2: rows i=1,2 via Strassen on 2x2x2 blocks (56 terms)
    # Split j into 2 blocks of 2: [0,1] and [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # For each (j_block, k_block): 7 Strassen multiplications

    # Strassen 2x2x2 factor matrices (7 multiplications)
    # A[i,j] with i,j in {0,1}, B[j,k] with j,k in {0,1}, C[i,k] with i,k in {0,1}
    # Local index for U: i*2+j -> 0,1,2,3
    # Local index for V: j*2+k -> 0,1,2,3
    # Local index for W: k*2+i -> 0,1,2,3  (trace convention W[k*n+i])

    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B00)
    # M4 = A11 (B01-B11)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    strassen_U = [
        [1, 0, 1, 1],    # M1: A00 + A11
        [0, 0, 1, 1],    # M2: A10 + A11
        [1, 0, 0, 0],    # M3: A00
        [0, 0, 0, 1],    # M4: A11
        [1, 1, 0, 0],    # M5: A00 + A01
        [-1, 0, 1, 0],   # M6: A10 - A00
        [0, 1, 0, -1],   # M7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],    # M1: B00 + B11
        [1, 0, 0, 0],    # M2: B00
        [0, 1, -1, 0],   # M3: B10 - B00
        [0, 1, 0, -1],   # M4: B01 - B11
        [0, 0, 0, 1],    # M5: B11
        [1, 1, 0, 0],    # M6: B00 + B01
        [0, 0, 1, 1],    # M7: B10 + B11
    ]
    strassen_W = [
        [1, 0, 0, 1],    # M1: C00 + C11
        [0, 0, 1, -1],   # M2: C10 - C00
        [0, 1, 0, 1],    # M3: C01 + C11
        [1, 0, 1, 0],    # M4: C00 + C10
        [-1, 1, 0, 0],   # M5: C01 - C11
        [0, 0, 0, 1],    # M6: C10
        [1, 0, 0, 0],    # M7: C00
    ]

    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                # U: local (li, lj) -> global i=1+li, j=j0+lj, row = i*m+j
                for li in range(2):
                    for lj in range(2):
                        val = strassen_U[t][li * 2 + lj]
                        if val != 0:
                            gi = 1 + li
                            gj = j0 + lj
                            U[gi * m + gj][col] += val

                # V: local (lj, lk) -> global j=j0+lj, k=k0+lk, row = j*p+k
                for lj in range(2):
                    for lk in range(2):
                        val = strassen_V[t][lj * 2 + lk]
                        if val != 0:
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk][col] += val

                # W: local (lk, li) -> global k=k0+lk, i=1+li, row = k*n+i
                for lk in range(2):
                    for li in range(2):
                        val = strassen_W[t][lk * 2 + li]
                        if val != 0:
                            gk = k0 + lk
                            gi = 1 + li
                            W[gk * n + gi][col] += val

                col += 1

        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
