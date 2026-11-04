import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _strassen_block():
    """Return the 7 Strassen terms for 2x2x2 as lists of (u_idx, v_idx, w_idx, coeff).
    u_idx = i*2+j (i,j in 0..1), v_idx = j*2+k, w_idx = k*2+i
    """
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append([
        ((0,0), 1), ((1,1), 1),  # u: (i,j)
        ((0,0), 1), ((1,1), 1),  # v: (j,k)
        ((0,0), 1), ((1,1), 1),  # w: (k,i)
    ])
    # M2 = (A10+A11)B00
    terms.append([
        ((1,0), 1), ((1,1), 1),
        ((0,0), 1),
        ((0,1), 1), ((1,1), -1),
    ])
    # M3 = A00(B01-B11)
    terms.append([
        ((0,0), 1),
        ((0,1), 1), ((1,1), -1),
        ((1,0), 1), ((1,1), 1),
    ])
    # M4 = A11(B10-B00)
    terms.append([
        ((1,1), 1),
        ((1,0), 1), ((0,0), -1),
        ((0,0), 1), ((1,0), 1),
    ])
    # M5 = (A00+A01)B11
    terms.append([
        ((0,0), 1), ((0,1), 1),
        ((1,1), 1),
        ((0,0), -1), ((1,0), 1),
    ])
    # M6 = (A10-A00)(B00+B01)
    terms.append([
        ((1,0), 1), ((0,0), -1),
        ((0,0), 1), ((0,1), 1),
        ((1,1), 1),
    ])
    # M7 = (A01-A11)(B10+B11)
    terms.append([
        ((0,1), 1), ((1,1), -1),
        ((1,0), 1), ((1,1), 1),
        ((0,0), 1),
    ])
    return terms

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    strassen_terms = _strassen_block()
    col = 0
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            for term in strassen_terms:
                u_coeffs, v_coeffs, w_coeffs = term
                for (i, j), val in u_coeffs:
                    if val != 0:
                        U[i*m + j][col] += val
                for (j, k), val in v_coeffs:
                    if val != 0:
                        V[j*p + k][col] += val
                for (k, i), val in w_coeffs:
                    if val != 0:
                        W[k*n + i][col] += val
                col += 1
    
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
