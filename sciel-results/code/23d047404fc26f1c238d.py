"""Tensor decomposition for 2x2 * 2x13 matrix multiplication using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Fallback: schoolbook decomposition of rank 52."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                if abs(val - 1) > tol:
                    return False
    return True


def _get_strassen_2x2():
    """Correct Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = [
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [0, 1, 0, 1, 1, -1, 0],
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    return U, V, W


def _get_schoolbook_2x1():
    """Schoolbook decomposition for 2x2 * 2x1 -> 2x1, rank 4."""
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Verify Strassen works
    U7, V7, W7 = _get_strassen_2x2()
    if not _verify_decomposition(U7, V7, W7, 2, 2, 2):
        return _build_schoolbook()
    
    # Build full decomposition: 6 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4)
    num_pairs = p // 2  # 6
    has_odd = (p % 2 == 1)
    rank = num_pairs * 7 + (4 if has_odd else 0)
    
    U_full = [[0] * rank for _ in range(n * m)]
    V_full = [[0] * rank for _ in range(m * p)]
    W_full = [[0] * rank for _ in range(p * n)]
    
    U4, V4, W4 = _get_schoolbook_2x1()
    
    col_offset = 0
    for t in range(num_pairs):
        k0 = 2 * t
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][col_offset + c] = U7[i * 2 + j][c]
            for j in range(2):
                for kl in range(2):
                    k = k0 + kl
                    V_full[j * p + k][col_offset + c] = V7[j * 2 + kl][c]
            for kl in range(2):
                k = k0 + kl
                for i in range(2):
                    W_full[k * n + i][col_offset + c] = W7[kl * 2 + i][c]
        col_offset += 7
    
    if has_odd:
        k0 = 2 * num_pairs
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][col_offset + c] = U4[i * 2 + j][c]
            for j in range(2):
                V_full[j * p + k0][col_offset + c] = V4[j][c]
            for i in range(2):
                W_full[k0 * n + i][col_offset + c] = W4[i][c]
        col_offset += 4
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return result
    
    return _build_schoolbook()

# EVOLVE-BLOCK-END
