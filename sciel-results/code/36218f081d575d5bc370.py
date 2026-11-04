import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _winograd_444_G():
    """Winograd G matrix for F(4,4,4)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

def _add_winograd_444(U, V, W, col, a_base, b_base, c_base):
    """Add Winograd F(4,4,4) rank-16 terms. Bases are global row-major linear indices."""
    G = _winograd_444_G()
    for a in range(4):
        for b in range(4):
            r = col + a * 4 + b
            for i in range(4):
                for j in range(4):
                    U[a_base + i * 4 + j][r] += G[i, a] * G[j, b]
            for j in range(4):
                for k in range(4):
                    V[b_base + j * 4 + k][r] += G[j, a] * G[k, b]
            for k in range(4):
                for i in range(4):
                    W[c_base + k * 4 + i][r] += G[i, b] * G[k, a]
    return col + 16

def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1. Schoolbook, rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col

def _verify(U, V, W, n, m, p):
    """Return True iff the tensor decomposition is exact."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _build_rank48(n, m, p):
    """Winograd(16) + Winograd(16) + matrix-vector(16) = 48.
    Split 4x4x9 into 4x4x4 (Winograd, 16) + 4x4x4 (Winograd, 16) + 4x4x1 (schoolbook, 16)."""
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    col = _add_winograd_444(U, V, W, col, 0, 0, 0)
    col = _add_winograd_444(U, V, W, col, 0, 4, 4)
    col = _add_matrix_vector_block(U, V, W, col, 0, 8, 8, 4, 4, 1)

    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_4x9(n, m, p):
    """
    Use Strassen's 7-multiplication algorithm to multiply 4x4 by 4x9.
    Split into 2x2 blocks of 2x2 matrices.
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    C = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2
    
    Strassen:
    M1 = (A11+A22)*(B11+B23)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B23-B13)
    M5 = (A11+A12)*B23
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B22+B23)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C13 = M2 + M4  (wait, let me recheck)
    
    Actually standard Strassen for C = AB:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M6 + M7
    
    For rectangular, B has 3 column blocks. Let's define:
    B = [B_left, B_mid, B_right] where B_left is 4x4, B_mid is 4x4, B_right is 4x1?
    
    No, let's use 2x2 block structure with 2x2 blocks.
    A is 4x4, so 2x2 blocks of 2x2.
    B is 4x9. Let's split B into 4x4, 4x4, 4x1? That's awkward.
    
    Alternative: Use the known result that R(4,4,9) <= R(4,4,4) + R(4,4,4) + R(4,4,1) = 16+16+16=48.
    
    Let's try a different approach: embed 4x4x9 as a sub-tensor of 8x8x8.
    But that gives high rank.
    
    Let's stick with the 48-rank construction but verify it works.
    """
    return _build_rank48(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    res = _build_rank48(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    return _build_schoolbook(n, m, p)
