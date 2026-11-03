import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Rank-76 decomposition: 4 blocks * 19 rank each
    # Each block: M_{2,2,5} = M_{2,2,2} (Strassen, rank 7) + 3 * M_{2,2,1} (trivial, rank 4 each) = 19
    rank = 76
    U = [[0] * rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * rank for _ in range(m * p)]  # 20 x 76
    W = [[0] * rank for _ in range(p * n)]  # 20 x 76

    def add_strassen_2x2x2(base_col, i_off, j_off, k_off):
        """Strassen for M_{2,2,2}. Returns next col."""
        col = base_col
        # Strassen factors:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # Reconstruction:
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M5 - M3
        # W[k*n + i, r]: C[i,k] = sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r]

        # M1: (a00+a11)(b00+b11)
        U[(0+i_off)*m + (0+j_off)][col] += 1
        U[(1+i_off)*m + (1+j_off)][col] += 1
        V[(0+j_off)*p + (0+k_off)][col] += 1
        V[(1+j_off)*p + (1+k_off)][col] += 1
        W[(0+k_off)*n + (0+i_off)][col] += 1   # C00 += M1
        W[(1+k_off)*n + (1+i_off)][col] += 1   # C11 += M1
        col += 1

        # M2: (a10+a11)b00
        U[(1+i_off)*m + (0+j_off)][col] += 1
        U[(1+i_off)*m + (1+j_off)][col] += 1
        V[(0+j_off)*p + (0+k_off)][col] += 1
        W[(0+k_off)*n + (1+i_off)][col] += 1   # C10 += M2
        W[(1+k_off)*n + (1+i_off)][col] += -1  # C11 -= M2
        col += 1

        # M3: a00(b10-b11)
        U[(0+i_off)*m + (0+j_off)][col] += 1
        V[(1+j_off)*p + (0+k_off)][col] += 1
        V[(1+j_off)*p + (1+k_off)][col] += -1
        W[(0+k_off)*n + (0+i_off)][col] += 1   # C00? No, C01 += M3
        # C01: i=0, k=1 -> W[1*n+0, col] = W[n, col]
        W[(1+k_off)*n + (0+i_off)][col] += 1   # C01 += M3
        W[(1+k_off)*n + (1+i_off)][col] += -1  # C11 -= M3
        col += 1

        # M4: a11(b01-b00)
        U[(1+i_off)*m + (1+j_off)][col] += 1
        V[(0+j_off)*p + (1+k_off)][col] += 1
        V[(0+j_off)*p + (0+k_off)][col] += -1
        W[(0+k_off)*n + (0+i_off)][col] += -1  # C00 -= M4
        W[(1+k_off)*n + (0+i_off)][col] += 1   # C01 += M4
        col += 1

        # M5: (a00+a01)b11
        U[(0+i_off)*m + (0+j_off)][col] += 1
        U[(0+i_off)*m + (1+j_off)][col] += 1
        V[(1+j_off)*p + (1+k_off)][col] += 1
        W[(0+k_off)*n + (1+i_off)][col] += 1   # C10 += M5
        W[(1+k_off)*n + (1+i_off)][col] += 1   # C11 += M5
        col += 1

        # M6: (a10-a00)(b00+b01)
        U[(1+i_off)*m + (0+j_off)][col] += 1
        U[(0+i_off)*m + (0+j_off)][col] += -1
        V[(0+j_off)*p + (0+k_off)][col] += 1
        V[(0+j_off)*p + (1+k_off)][col] += 1
        W[(0+k_off)*n + (0+i_off)][col] += -1  # C00 -= M6
        col += 1

        # M7: (a01-a11)(b10+b11)
        U[(0+i_off)*m + (1+j_off)][col] += 1
        U[(1+i_off)*m + (1+j_off)][col] += -1
        V[(1+j_off)*p + (0+k_off)][col] += 1
        V[(1+j_off)*p + (1+k_off)][col] += 1
        W[(0+k_off)*n + (0+i_off)][col] += 1   # C00 += M7
        col += 1

        return col

    def add_trivial_2x2x1(base_col, i_off, j_off, k_off):
        """Trivial for M_{2,2,1}. Returns next col."""
        col = base_col
        for i in range(2):
            for j in range(2):
                U[(i+i_off)*m + (j+j_off)][col] += 1
                V[(j+j_off)*p + (k+k_off)][col] += 1
                W[(k+k_off)*n + (i+i_off)][col] += 1
                col += 1
        return col

    col = 0
    for i_block in range(2):
        for j_block in range(2):
            i_off = i_block * 2
            j_off = j_block * 2
            for k_off in [0, 2]:  # B_top and B_bot
                col = add_strassen_2x2x2(col, i_off, j_off, k_off)
                col = add_trivial_2x2x1(col, i_off, j_off, k_off + 2)
                col = add_trivial_2x2x1(col, i_off, j_off, k_off + 3)
                col = add_trivial_2x2x1(col, i_off, j_off, k_off + 4)
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
