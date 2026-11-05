import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 150.0

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T_true, atol=1e-6):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return bool(np.allclose(T_recon, T_true, atol=atol))

def _snap_half(x):
    return np.round(2.0 * x) / 2.0

def _snap_all(U, V, W):
    return _snap_half(U), _snap_half(V), _snap_half(W)

# Strassen 2x2x2 term templates
_M1_A = {(0,0):1, (1,1):1}
_M1_B = {(0,0):1, (1,1):1}
_M1_C = {(0,0):1, (1,1):1}
_M2_A = {(1,0):1, (1,1):1}
_M2_B = {(0,0):1}
_M2_C = {(1,0):1}
_M3_A = {(0,0):1}
_M3_B = {(1,0):1, (1,1):-1}
_M3_C = {(0,1):1, (1,1):1}
_M4_A = {(1,1):1}
_M4_B = {(0,0):1, (0,1):-1}
_M4_C = {(0,0):1, (1,0):1}
_M5_A = {(0,0):1, (0,1):1}
_M5_B = {(1,1):1}
_M5_C = {(0,1):1, (0,0):-1}
_M6_A = {(1,0):1, (0,0):-1}
_M6_B = {(0,0):1, (0,1):1}
_M6_C = {(1,1):1}
_M7_A = {(0,1):1, (1,1):-1}
_M7_B = {(1,0):1, (1,1):1}
_M7_C = {(0,0):1, (1,1):-1}
_STRASSEN = [(_M1_A,_M1_B,_M1_C),(_M2_A,_M2_B,_M2_C),(_M3_A,_M3_B,_M3_C),
             (_M4_A,_M4_B,_M4_C),(_M5_A,_M5_B,_M5_C),(_M6_A,_M6_B,_M6_C),
             (_M7_A,_M7_B,_M7_C)]

def _add_strassen(U, V, W, col, i_base, j_base, k_base, n):
    for (a_c, b_c, c_c) in _STRASSEN:
        for (r, c), val in a_c.items():
            U[i_base + r, j_base + c, col] += val
        for (r, c), val in b_c.items():
            V[j_base + r, k_base + c, col] += val
        for (r, c), val in c_c.items():
            W[(k_base + c) * n + (i_base + r), col] += val
        col += 1
    return col

def _build_hybrid_strassen(n, m, p):
    # 3x4x8 = (2x4x8) + (1x4x8)
    # 2x4x8 = (2x4x4) + (2x4x4)
    # 2x4x4 = (2x2x4) + (2x2x4)
    # 2x2x4 = (2x2x2) + (2x2x2) = 7+7 = 14
    # 2x4x4 = 28
    # 2x4x8 = 56
    # 1x4x8 schoolbook = 32
    # total = 88
    nm, mp, pn = n * m, m * p, p * n
    rank = 56 + 32
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    # 2x4x8 part (i=0,1)
    for k_block in range(2):
        k_start = k_block * 4
        for j_block in range(2):
            j_start = j_block * 2
            for k_sub in range(2):
                k_start2 = k_start + k_sub * 2
                col = _add_strassen(U, V, W, col, 0, j_start, k_start2, n)
    # 1x4x8 part (i=2) schoolbook
    for i in [2]:
        for j in range(4):
            for k in range(8):
                U[i*4 + j, col] = 1
                V[j*8 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    return U, V, W, rank

def _build_schoolbook(n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    for col in range(rank):
        k = col % p
        j = (col // p) % m
        i = col // (p * m)
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
    return U, V, W, rank

def construct():
    n, m, p = 3, 4, 8
    T_true = _build_true_tensor(n, m, p)
    
    # Build hybrid Strassen decomposition (rank 88)
    U, V, W, rank = _build_hybrid_strassen(n, m, p)
    
    if not _verify(U, V, W, T_true):
        # Fallback to schoolbook
        U, V, W, rank = _build_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
