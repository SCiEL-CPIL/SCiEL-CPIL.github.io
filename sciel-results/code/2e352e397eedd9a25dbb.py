import time
from functools import lru_cache
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill

def _check_time():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2, rank 7."""
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [
        [1, 0, 0, 1],   # a+d
        [0, 0, 1, 1],   # c+d
        [1, 0, 0, 0],   # a
        [0, 0, 0, 1],   # d
        [1, 1, 0, 0],   # a+b
        [-1, 0, 1, 0],  # c-a
        [0, 1, 0, -1]   # b-d
    ]
    V = [
        [1, 0, 0, 1],   # e+h
        [1, 0, 0, 0],   # e
        [0, 1, 0, -1],  # f-h
        [-1, 0, 1, 0],  # g-e
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # e+f
        [0, 0, 1, 1]    # g+h
    ]
    # W convention: W[k*m+i, r] where output C is m x p
    # For 2x2: C11 at (0,0), C12 at (0,1), C21 at (1,0), C22 at (1,1)
    # W index: k*m+i where k is column of C, i is row of C
    # C11: k=0, i=0 -> W[0]
    # C12: k=1, i=0 -> W[2]  (k*m+i = 1*2+0 = 2)
    # C21: k=0, i=1 -> W[1]
    # C22: k=1, i=1 -> W[3]
    
    # M1 contributes to C11 and C22
    # M4 contributes to C11 and C21
    # M5 contributes to C11 (negative) and C12
    # M7 contributes to C11 (negative)
    # M3 contributes to C12 and C22
    # M2 contributes to C21 and C22 (negative)
    # M6 contributes to C22
    
    W = [
        [1, 1, -1, -1, 0, 0, 0],   # C11 = M1 + M4 - M5 - M7
        [0, 0, 0, 0, 1, 0, 0],     # C12 = M3 + M5... wait
        [0, 1, 1, 0, 0, 0, 0],     # C21 = M2 + M4
        [1, -1, 0, 0, 0, 1, 0]     # C22 = M1 - M2 + M3 + M6
    ]
    # Let me redo W carefully.
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W = [
        [1, 0, 0, 1, -1, 0, -1],   # C11
        [0, 0, 1, 0, 1, 0, 0],     # C12
        [0, 1, 0, 1, 0, 0, 0],     # C21
        [1, -1, 1, 0, 0, 1, 0]     # C22
    ]
    return (U, V, W)

def _schoolbook(m, n, p):
    """Return schoolbook decomposition for m x n * n x p."""
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

def _combine(U1, V1, W1, m1, n1, p1, offset_m, offset_n, offset_p, m, n, p):
    """Add a sub-decomposition to the full-size tensors."""
    U_new = []
    V_new = []
    W_new = []
    for r in range(len(U1)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m1):
            for jj in range(n1):
                u_row[(offset_m + ii) * n + (offset_n + jj)] = U1[r][ii * n1 + jj]
        for jj in range(n1):
            for kk in range(p1):
                v_row[(offset_n + jj) * p + (offset_p + kk)] = V1[r][jj * p1 + kk]
        for kk in range(p1):
            for ii in range(m1):
                w_row[(offset_p + kk) * m + (offset_m + ii)] = W1[r][kk * m1 + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return (U_new, V_new, W_new)

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
        return _strassen_2x2x2()
    
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
                    # 8 sub-decompositions
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        # A11*B11: m1×n1 * n1×p1
                        u_sub, v_sub, w_sub = _combine(U1, V1, W1, m1, n1, p1, 0, 0, 0, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A12*B21: m1×n2 * n2×p1
                        u_sub, v_sub, w_sub = _combine(U3, V3, W3, m1, n2, p1, 0, n1, 0, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A11*B12: m1×n1 * n1×p2
                        u_sub, v_sub, w_sub = _combine(U2, V2, W2, m1, n1, p2, 0, 0, p1, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A12*B22: m1×n2 * n2×p2
                        u_sub, v_sub, w_sub = _combine(U4, V4, W4, m1, n2, p2, 0, n1, p1, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A21*B11: m2×n1 * n1×p1
                        u_sub, v_sub, w_sub = _combine(U5, V5, W5, m2, n1, p1, m1, 0, 0, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A21*B12: m2×n1 * n1×p2
                        u_sub, v_sub, w_sub = _combine(U6, V6, W6, m2, n1, p2, m1, 0, p1, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A22*B21: m2×n2 * n2×p1
                        u_sub, v_sub, w_sub = _combine(U7, V7, W7, m2, n2, p1, m1, n1, 0, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        # A22*B22: m2×n2 * n2×p2
                        u_sub, v_sub, w_sub = _combine(U8, V8, W8, m2, n2, p2, m1, n1, p1, m, n, p)
                        U_new.extend(u_sub); V_new.extend(v_sub); W_new.extend(w_sub)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
