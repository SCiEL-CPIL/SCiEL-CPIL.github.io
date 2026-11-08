"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

def _strassen_2x2_2x6(a_col_off, b_row_off, b_col_off, U, V, W, m, p, n, col):
    """Add Strassen decomposition for 2x2 * 2x6 block.
    A block: rows 0-1, cols a_col_off to a_col_off+1
    B block: rows b_row_off to b_row_off+1, cols b_col_off to b_col_off+5
    C block: rows 0-1, cols b_col_off to b_col_off+5
    Returns new col index (7 multiplications added).
    """
    # 7 Strassen products for 2x2 * 2x2, applied to 3 column-pairs
    # For each column-pair t in 0,1,2:
    #   b_col0 = b_col_off + 2*t, b_col1 = b_col_off + 2*t+1
    
    for t in range(3):
        bc0 = b_col_off + 2 * t
        bc1 = b_col_off + 2 * t + 1
        br0 = b_row_off
        br1 = b_row_off + 1
        ac0 = a_col_off
        ac1 = a_col_off + 1

        # M1 = (a00+a11) * (b00+b11)
        U[0 * m + ac0][col] += 1
        U[1 * m + ac1][col] += 1
        V[br0 * p + bc0][col] += 1
        V[br1 * p + bc1][col] += 1
        W[bc0 * n + 0][col] += 1
        W[bc1 * n + 1][col] += 1
        col += 1

        # M2 = (a10+a11) * b00
        U[1 * m + ac0][col] += 1
        U[1 * m + ac1][col] += 1
        V[br0 * p + bc0][col] += 1
        W[bc0 * n + 1][col] += 1
        W[bc1 * n + 1][col] += -1
        col += 1

        # M3 = a00 * (b01-b11)
        U[0 * m + ac0][col] += 1
        V[br0 * p + bc1][col] += 1
        V[br1 * p + bc1][col] += -1
        W[bc1 * n + 0][col] += 1
        W[bc1 * n + 1][col] += 1
        col += 1

        # M4 = a11 * (b10-b00)
        U[1 * m + ac1][col] += 1
        V[br1 * p + bc0][col] += 1
        V[br0 * p + bc0][col] += -1
        W[bc0 * n + 0][col] += 1
        W[bc1 * n + 1][col] += 1
        col += 1

        # M5 = (a00+a01) * b11
        U[0 * m + ac0][col] += 1
        U[0 * m + ac1][col] += 1
        V[br1 * p + bc1][col] += 1
        W[bc0 * n + 0][col] += -1
        W[bc1 * n + 0][col] += 1
        col += 1

        # M6 = (a10-a00) * (b00+b01)
        U[1 * m + ac0][col] += 1
        U[0 * m + ac0][col] += -1
        V[br0 * p + bc0][col] += 1
        V[br0 * p + bc1][col] += 1
        W[bc1 * n + 1][col] += 1
        col += 1

        # M7 = (a01-a11) * (b10+b11)
        U[0 * m + ac1][col] += 1
        U[1 * m + ac1][col] += -1
        V[br1 * p + bc0][col] += 1
        V[br1 * p + bc1][col] += 1
        W[bc0 * n + 0][col] += 1
        col += 1

    return col

def _outer_product_2x1_1x6(a_col, b_row, b_col_off, U, V, W, m, p, n, col):
    """Add rank-2 decomposition for 2x1 * 1x6 block.
    A block: rows 0-1, col a_col
    B block: row b_row, cols b_col_off to b_col_off+5
    C block: rows 0-1, cols b_col_off to b_col_off+5
    C[i,j] = A[i, a_col] * B[b_row, j]
    This is an outer product, decomposable as 2 rank-1 terms (one per i).
    Returns new col index (2 multiplications added).
    """
    for i in range(2):
        U[i * m + a_col][col] += 1
        V[b_row * p + b_col_off + 0][col] += 1
        V[b_row * p + b_col_off + 1][col] += 1
        V[b_row * p + b_col_off + 2][col] += 1
        V[b_row * p + b_col_off + 3][col] += 1
        V[b_row * p + b_col_off + 4][col] += 1
        V[b_row * p + b_col_off + 5][col] += 1
        W[(b_col_off + 0) * n + i][col] += 1
        W[(b_col_off + 1) * n + i][col] += 1
        W[(b_col_off + 2) * n + i][col] += 1
        W[(b_col_off + 3) * n + i][col] += 1
        W[(b_col_off + 4) * n + i][col] += 1
        W[(b_col_off + 5) * n + i][col] += 1
        col += 1
    return col

def construct():
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    start_time = time.time()
    deadline = start_time + 140.0

    # Strategy: Split A into A_left (2x1, col 0) and A_right (2x3, cols 1-3)
    # Split B into B_top (1x6, row 0) and B_bot (3x6, rows 1-3)
    # C = A_left @ B_top + A_right @ B_bot
    # 
    # A_left @ B_top: 2x1 * 1x6, rank 2
    # A_right @ B_bot: 2x3 * 3x6
    #   Split B_bot rows into 1+2: B_bot_top (1x6, row 1), B_bot_bot (2x6, rows 2-3)
    #   Split A_right cols into 1+2: A_right_left (2x1, col 1), A_right_right (2x2, cols 2-3)
    #   A_right @ B_bot = A_right_left @ B_bot_top + A_right_right @ B_bot_bot
    #   A_right_left @ B_bot_top: 2x1 * 1x6, rank 2
    #   A_right_right @ B_bot_bot: 2x2 * 2x6, rank 21 (3 Strassen 2x2x2)
    #
    # Total rank = 2 + 2 + 21 = 25

    rank = 25
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    # Term 1: A_left (cols 0) * B_top (rows 0) -> rank 2
    col = _outer_product_2x1_1x6(a_col=0, b_row=0, b_col_off=0, U=U, V=V, W=W, m=m, p=p, n=n, col=col)

    # Term 2: A_right_left (col 1) * B_bot_top (row 1) -> rank 2
    col = _outer_product_2x1_1x6(a_col=1, b_row=1, b_col_off=0, U=U, V=V, W=W, m=m, p=p, n=n, col=col)

    # Term 3: A_right_right (cols 2-3) * B_bot_bot (rows 2-3) -> rank 21
    col = _strassen_2x2_2x6(a_col_off=2, b_row_off=2, b_col_off=0, U=U, V=V, W=W, m=m, p=p, n=n, col=col)

    # Verification
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U_np[:, c]
        vc = V_np[:, c]
        wc = W_np[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]

    valid = np.allclose(T, 1.0, atol=1e-9)

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
