import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    terms = []

    def strassen_2x2(i_off, j_off, k_off):
        """
        Generate 7 Strassen terms for a 2x2*2x2 block multiplication.
        A block at rows [i_off, i_off+1], cols [j_off, j_off+1]
        B block at rows [j_off, j_off+1], cols [k_off, k_off+1]
        C block at rows [i_off, i_off+1], cols [k_off, k_off+1]
        """
        result = []

        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        result.append((u[:], v[:], w[:]))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        result.append((u[:], v[:], w[:]))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        result.append((u[:], v[:], w[:]))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        result.append((u[:], v[:], w[:]))

        return result

    # 1. 4x4 * 4x4 part (k=0,1,2,3)
    for i_b in range(2):
        for j_b in range(2):
            for k_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = k_b * 2
                terms.extend(strassen_2x2(i_off, j_off, k_off))

    # 2. 4x4 * 4x2 part (k=4,5)
    # A (4x4) * B[:, 4:6] (4x2)
    # Decompose into 2x2 blocks
    # i_blocks: 0, 1 (rows 0-1, 2-3)
    # j_blocks: 0, 1 (cols 0-1, 2-3)
    # k_block: 4 (cols 4-5)
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            k_off = 4
            terms.extend(strassen_2x2(i_off, j_off, k_off))

    # 3. 4x4 * 4x1 part (k=6)
    # A (4x4) * B[:, 6] (4x1)
    # Decompose into 2x2 * 2x1 blocks
    for i_b in range(2):
        for j_b in range(2):
            i_off = i_b * 2
            j_off = j_b * 2
            k_off = 6
            
            # Term 1: A[i, j] * B[j, 6] -> C[i, 6]
            u = [0] * (n * m)
            u[i_off * m + j_off] = 1
            v = [0] * (m * p)
            v[j_off * p + k_off] = 1
            w = [0] * (p * n)
            w[k_off * n + i_off] = 1
            terms.append((u, v, w))

            # Term 2: A[i, j+1] * B[j+1, 6] -> C[i, 6]
            u = [0] * (n * m)
            u[i_off * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = 1
            w = [0] * (p * n)
            w[k_off * n + i_off] = 1
            terms.append((u, v, w))

            # Term 3: A[i+1, j] * B[j, 6] -> C[i+1, 6]
            u = [0] * (n * m)
            u[(i_off+1) * m + j_off] = 1
            v = [0] * (m * p)
            v[j_off * p + k_off] = 1
            w = [0] * (p * n)
            w[k_off * n + (i_off+1)] = 1
            terms.append((u, v, w))

            # Term 4: A[i+1, j+1] * B[j+1, 6] -> C[i+1, 6]
            u = [0] * (n * m)
            u[(i_off+1) * m + (j_off+1)] = 1
            v = [0] * (m * p)
            v[(j_off+1) * p + k_off] = 1
            w = [0] * (p * n)
            w[k_off * n + (i_off+1)] = 1
            terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
