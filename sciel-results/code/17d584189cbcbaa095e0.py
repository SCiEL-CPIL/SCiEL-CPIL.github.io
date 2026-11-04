"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1   # C00 += M1
    W[3,0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1   # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1   # C01 += M3
    W[3,2] = -1  # C11 -= M3
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1  # C00 -= M4
    W[2,3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # C00 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = 1   # C11 += M7
    
    return U, V, W

# EVOLVE-BLOCK-START

_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _check_time():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()

def _schoolbook(m, n, p):
    """Return (U, V, W) for m×n * n×p matrix multiplication using schoolbook."""
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
    return U, V, W

def _strassen_2x2x2_list():
    """Return (U, V, W) for 2x2x2 Strassen decomposition (rank 7) as lists."""
    U = []
    V = []
    W = []
    # M1 = (a+d)(e+h)
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 0, 0, 1])
    # M2 = (c+d)e
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, -1, 0, -1])
    # M3 = a(f-h)
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 1, 0, 1])
    # M4 = d(g-e)
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([1, 0, 1, 0])
    # M5 = (a+b)h
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 0, 0, 1])
    # M6 = (c-a)(e+f)
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1])
    # M7 = (b-d)(g+h)
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, 0])
    return U, V, W

def _decompose_rank(m, n, p):
    """Return the minimal rank for m×n * n×p via recursive block decomposition."""
    _check_time()
    if m == 0 or n == 0 or p == 0:
        return 0
    if m == 1 and n == 1 and p == 1:
        return 1
    if m == 2 and n == 2 and p == 2:
        return 7
    if m == 1:
        return n * p
    if p == 1:
        return m * n
    if n == 1:
        return m * p
    
    best = m * n * p
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                r = (_decompose_rank(m1, n1, p1) + _decompose_rank(m1, n1, p2) +
                     _decompose_rank(m1, n2, p1) + _decompose_rank(m1, n2, p2) +
                     _decompose_rank(m2, n1, p1) + _decompose_rank(m2, n1, p2) +
                     _decompose_rank(m2, n2, p1) + _decompose_rank(m2, n2, p2))
                if r < best:
                    best = r
    return best

@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    _check_time()
    
    if m == 0 or n == 0 or p == 0:
        return (), (), ()
    if m == 1 and n == 1 and p == 1:
        return ([[1]], [[1]], [[1]])
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2_list()
    
    if m == 1:
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
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * m + i][col] = 1
                col += 1
        return (U, V, W)
    
    # Pre-compute the optimal rank for this subproblem
    target_rank = _decompose_rank(m, n, p)
    
    # Try all possible splits, find one that achieves target_rank
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                _check_time()
                
                r1 = _decompose_rank(m1, n1, p1)
                r2 = _decompose_rank(m1, n1, p2)
                r3 = _decompose_rank(m1, n2, p1)
                r4 = _decompose_rank(m1, n2, p2)
                r5 = _decompose_rank(m2, n1, p1)
                r6 = _decompose_rank(m2, n1, p2)
                r7 = _decompose_rank(m2, n2, p1)
                r8 = _decompose_rank(m2, n2, p2)
                total_rank = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
                
                if total_rank != target_rank:
                    continue
                
                try:
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    U_new = []
                    V_new = []
                    W_new = []
                    
                    def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, off_m, off_n, off_p):
                        for r in range(len(U_s)):
                            u_row = [0] * (m * n)
                            v_row = [0] * (n * p)
                            w_row = [0] * (p * m)
                            for ii in range(m_i):
                                for jj in range(n_i):
                                    u_row[(off_m + ii) * n + (off_n + jj)] = U_s[r][ii * n_i + jj]
                            for jj in range(n_i):
                                for kk in range(p_i):
                                    v_row[(off_n + jj) * p + (off_p + kk)] = V_s[r][jj * p_i + kk]
                            for kk in range(p_i):
                                for ii in range(m_i):
                                    w_row[(off_p + kk) * m + (off_m + ii)] = W_s[r][kk * m_i + ii]
                            U_new.append(u_row)
                            V_new.append(v_row)
                            W_new.append(w_row)
                    
                    add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                    add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                    add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                    add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                    add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                    add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                    add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                    add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                    
                    return (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    # Fallback: schoolbook
    return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    try:
        U, V, W = _decompose(n, m, p)
        return {"U": U, "V": V, "W": W}
    except TimeoutError:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
