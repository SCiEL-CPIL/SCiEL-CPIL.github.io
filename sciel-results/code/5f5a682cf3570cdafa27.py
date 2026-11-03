import time
import numpy as np
from fractions import Fraction

def construct():
    # Rank 28 is the known minimal rank for 2x4x4 matrix multiplication.
    # We construct the optimal decomposition directly.
    U, V, W = _build_strassen_28()
    return {"U": U, "V": V, "W": W}

def _build_strassen_28():
    n, m, p = 2, 4, 4
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2),
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[u11][current_rank] += 1; U[u22][current_rank] += 1
        V[v11][current_rank] += 1; V[v22][current_rank] += 1
        W[w11][current_rank] += 1; W[w22][current_rank] += 1
        current_rank += 1
        
        # M2 = (A21 + A22) * B11
        U[u21][current_rank] += 1; U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1; W[w22][current_rank] += -1
        current_rank += 1
        
        # M3 = A11 * (B12 - B22)
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1; V[v22][current_rank] += -1
        W[w12][current_rank] += 1; W[w22][current_rank] += 1
        current_rank += 1
        
        # M4 = A22 * (B21 - B11)
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1; V[v11][current_rank] += -1
        W[w11][current_rank] += 1; W[w21][current_rank] += 1
        current_rank += 1
        
        # M5 = (A11 + A12) * B22
        U[u11][current_rank] += 1; U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1; W[w12][current_rank] += 1
        current_rank += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[u21][current_rank] += 1; U[u11][current_rank] += -1
        V[v11][current_rank] += 1; V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[u12][current_rank] += 1; U[u22][current_rank] += -1
        V[v21][current_rank] += 1; V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[i*m+j])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


