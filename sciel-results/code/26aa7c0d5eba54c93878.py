import time
from functools import lru_cache
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill


def _check_time():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()


def _transpose_matrix(mat):
    """Transpose a 2D list."""
    if not mat or not mat[0]:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[i][j] for i in range(rows)] for j in range(cols)]


def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7) with W[k*2+i, r] convention"""
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: each column r gives the linear combination of A elements for multiplication r
    # U has shape (m*n, rank) = (4, 7)
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a: coeff in M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],    # b: coeff in M5, M7
        [0, 1, 0, 0, 0, 1, 0],    # c: coeff in M2, M6
        [1, 0, 0, 1, 0, 0, -1]    # d: coeff in M1, M4, M7
    ]
    # V: each column r gives the linear combination of B elements
    # V has shape (n*p, rank) = (4, 7)
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # e: coeff in M1, M2, M4, M6
        [0, 0, 1, 0, 0, 0, 0],    # f: coeff in M3
        [0, 0, 0, 1, 0, 0, 1],    # g: coeff in M4, M7
        [1, 0, -1, 0, 1, 0, 0]    # h: coeff in M1, M3, M5
    ]
    # W: W[k*m+i, r] where k is col of B, i is row of A
    # W has shape (p*m, rank) = (4, 7)
    # C11 at (0,0), C12 at (0,1), C21 at (1,0), C22 at (1,1)
    # W index: k*m+i, so C11->0*2+0=0, C12->0*2+1=1, C21->1*2+0=2, C22->1*2+1=3
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11: M1, M4, -M5, M7
        [0, 0, 1, 0, 1, 0, 0],    # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0],    # C21: M2, M4
        [1, -1, 1, 0, 0, 1, 0]    # C22: M1, -M2, M3, M6
    ]
    return (U, V, W)


def _combine_decompositions(m, n, p, decomps):
    """
    Combine sub-decompositions into a full decomposition.
    decomps: list of (U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p)
    U_s: (m_i*n_i, rank_s), V_s: (n_i*p_i, rank_s), W_s: (p_i*m_i, rank_s)
    Returns: (U, V, W) with U: (m*n, rank), V: (n*p, rank), W: (p*m, rank)
    """
    total_rank = sum(len(d[0][0]) if d[0] else 0 for d in decomps)
    U_new = [[0] * total_rank for _ in range(m * n)]
    V_new = [[0] * total_rank for _ in range(n * p)]
    W_new = [[0] * total_rank for _ in range(p * m)]
    
    rank_offset = 0
    for (U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p) in decomps:
        rank_s = len(U_s[0]) if U_s else 0
        for ii in range(m_i):
            for jj in range(n_i):
                u_idx = (offset_m + ii) * n + (offset_n + jj)
                u_s_idx = ii * n_i + jj
                for r in range(rank_s):
                    U_new[u_idx][rank_offset + r] = U_s[u_s_idx][r]
        for jj in range(n_i):
            for kk in range(p_i):
                v_idx = (offset_n + jj) * p + (offset_p + kk)
                v_s_idx = jj * p_i + kk
                for r in range(rank_s):
                    V_new[v_idx][rank_offset + r] = V_s[v_s_idx][r]
        for kk in range(p_i):
            for ii in range(m_i):
                w_idx = (offset_p + kk) * m + (offset_m + ii)
                w_s_idx = kk * m_i + ii
                for r in range(rank_s):
                    W_new[w_idx][rank_offset + r] = W_s[w_s_idx][r]
        rank_offset += rank_s
    
    return (U_new, V_new, W_new), total_rank


@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    _check_time()
    
    # Base case: empty
    if m == 0 or n == 0 or p == 0:
        return ([], [], [])
    
    # Degenerate cases
    if m == 1 and n == 1 and p == 1:
        return ([[1]], [[1]], [[1]])
        
    if m == 1:
        # Rank n*p
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for j in range(n):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return (U, V, W)
        
    if p == 1:
        # Rank m*n
        rank = m * n
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n)]
        W = [[0] * rank for _ in range(m)]
        col = 0
        for i in range(m):
            for j in range(n):
                U[i * n + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return (U, V, W)
        
    if n == 1:
        # Rank m*p
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(m * p)]
        col = 0
        for i in range(m):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * m + i][col] = 1
                col += 1
        return (U, V, W)

    # Special case: Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        _check_time()
        for n1 in range(1, n):
            n2 = n - n1
            _check_time()
            for p1 in range(1, p):
                p2 = p - p1
                _check_time()
                try:
                    # 8 sub-decompositions (cached)
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1[0]) if U1 else 0 + len(U2[0]) if U2 else 0 +
                                 len(U3[0]) if U3 else 0 + len(U4[0]) if U4 else 0 +
                                 len(U5[0]) if U5 else 0 + len(U6[0]) if U6 else 0 +
                                 len(U7[0]) if U7 else 0 + len(U8[0]) if U8 else 0)
                    if total_rank < best_rank:
                        best_rank = total_rank
                        decomps = [
                            (U1, V1, W1, m1, n1, p1, 0, 0, 0),
                            (U3, V3, W3, m1, n2, p1, 0, n1, 0),
                            (U2, V2, W2, m1, n1, p2, 0, 0, p1),
                            (U4, V4, W4, m1, n2, p2, 0, n1, p1),
                            (U5, V5, W5, m2, n1, p1, m1, 0, 0),
                            (U6, V6, W6, m2, n1, p2, m1, 0, p1),
                            (U7, V7, W7, m2, n2, p1, m1, n1, 0),
                            (U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        ]
                        best, _ = _combine_decompositions(m, n, p, decomps)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        # Schoolbook
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return (U, V, W)


def construct():
    global _start_time
    _start_time = time.time()
    # Clear cache to start fresh
    _decompose.cache_clear()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
