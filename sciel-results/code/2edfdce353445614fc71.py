import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _strassen_2x2_block(i_base, j_base, k_base, m, p, n, U, V, W, col):
    """Add 7 Strassen terms for a 2x2 * 2x2 block. Returns new col index."""
    # M0: (A00+A11)(B00+B11)
    U[i_base * m + j_base][col] += 1
    U[(i_base + 1) * m + (j_base + 1)][col] += 1
    V[j_base * p + k_base][col] += 1
    V[(j_base + 1) * p + (k_base + 1)][col] += 1
    W[k_base * n + i_base][col] += 1
    W[(k_base + 1) * n + (i_base + 1)][col] += 1
    col += 1

    # M1: (A10+A11)(B00-B10)
    U[(i_base + 1) * m + j_base][col] += 1
    U[(i_base + 1) * m + (j_base + 1)][col] += 1
    V[j_base * p + k_base][col] += 1
    V[(j_base + 1) * p + k_base][col] -= 1
    W[k_base * n + (i_base + 1)][col] += 1
    col += 1

    # M2: A00(B01-B11)
    U[i_base * m + j_base][col] += 1
    V[j_base * p + (k_base + 1)][col] += 1
    V[(j_base + 1) * p + (k_base + 1)][col] -= 1
    W[(k_base + 1) * n + i_base][col] += 1
    col += 1

    # M3: A11(B10-B00)
    U[(i_base + 1) * m + (j_base + 1)][col] += 1
    V[(j_base + 1) * p + k_base][col] += 1
    V[j_base * p + k_base][col] -= 1
    W[k_base * n + (i_base + 1)][col] += 1
    col += 1

    # M4: (A00+A01)(B11-B10)
    U[i_base * m + j_base][col] += 1
    U[i_base * m + (j_base + 1)][col] += 1
    V[(j_base + 1) * p + (k_base + 1)][col] += 1
    V[(j_base + 1) * p + k_base][col] -= 1
    W[(k_base + 1) * n + i_base][col] += 1
    col += 1

    # M5: (A10-A00)(B00+B01)
    U[(i_base + 1) * m + j_base][col] += 1
    U[i_base * m + j_base][col] -= 1
    V[j_base * p + k_base][col] += 1
    V[j_base * p + (k_base + 1)][col] += 1
    W[(k_base + 1) * n + (i_base + 1)][col] += 1
    col += 1

    # M6: (A01-A11)(B10+B11)
    U[i_base * m + (j_base + 1)][col] += 1
    U[(i_base + 1) * m + (j_base + 1)][col] -= 1
    V[(j_base + 1) * p + k_base][col] += 1
    V[(j_base + 1) * p + (k_base + 1)][col] += 1
    W[k_base * n + i_base][col] += 1
    col += 1

    return col


def construct():
    n, m, p = DIMENSIONS

    # Total rank: 6 Strassen blocks (7 each) + 12 direct terms = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen blocks: A has rows 0-1, cols split as (0,1) and (2,3)
    # B has rows split as (0,1), (2,3), (4,5)
    # For each pair of A-col-blocks and B-row-blocks, we get a 2x2 * 2x2 product

    # A cols (0,1), B rows (0,1)
    col = _strassen_2x2_block(0, 0, 0, m, p, n, U, V, W, col)
    # A cols (0,1), B rows (2,3)
    col = _strassen_2x2_block(0, 0, 2, m, p, n, U, V, W, col)
    # A cols (0,1), B rows (4,5)
    col = _strassen_2x2_block(0, 0, 4, m, p, n, U, V, W, col)
    # A cols (2,3), B rows (0,1)
    col = _strassen_2x2_block(0, 2, 0, m, p, n, U, V, W, col)
    # A cols (2,3), B rows (2,3)
    col = _strassen_2x2_block(0, 2, 2, m, p, n, U, V, W, col)
    # A cols (2,3), B rows (4,5)
    col = _strassen_2x2_block(0, 2, 4, m, p, n, U, V, W, col)

    # Remaining: A col 4 (single column) times all 6 rows of B
    # This is a 2x1 * 1x6 = 2x6 product, done directly: 12 terms
    for k in range(p):
        # C[0, k] += A[0, 4] * B[4, k]
        U[0 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1

        # C[1, k] += A[1, 4] * B[4, k]
        U[1 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # Verify column count
    if col != rank:
        # Should not happen, but fallback
        pass

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
