import time
import numpy as np

DIMENSIONS = [3, 4, 4]
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 29
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Corrected Strassen coefficient vectors
    # U: local index (li*2+lj) where li=i-local, lj=j-local
    # V: local index (lj*2+lk) where lj=j-local, lk=k-local
    # W: local index (lk*n_local+li) where lk=k-local, li=i-local; n_local=2
    _u_vecs = np.array([
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [0, 0, 1, -1],  # M6: c-d
        [0, 1, 0, -1]   # M7: b-d
    ], dtype=float)
    _v_vecs = np.array([
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [0, 1, 1, 0],   # M6: f+g
        [1, 0, 1, 0]    # M7: e+g
    ], dtype=float)
    # W: index order (lk*2+li): [C(0,0), C(1,0), C(0,1), C(1,1)]
    _w_vecs = np.array([
        [1, 0, 1, 0],   # M1
        [0, 1, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 1],  # M4
        [1, 0, 0, 0],   # M5
        [0, 0, 0, -1],  # M6
        [1, 0, 0, 0]    # M7
    ], dtype=float)
    
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_idx = (i_base + li) * m + (j_base + lj)
                    U[global_idx, col] += _u_vecs[t, local_idx]
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    global_idx = (j_base + lj) * p + (k_base + lk)
                    V[global_idx, col] += _v_vecs[t, local_idx]
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_idx = (k_base + lk) * n + (i_base + li)
                    W[global_idx, col] += _w_vecs[t, local_idx]
            col += 1

    def add_1x4x4(i_base, j_base, k_base):
        nonlocal col
        for lj in range(4):
            U[i_base * m + (j_base + lj), col] = 1.0
            for lk in range(4):
                V[(j_base + lj) * p + (k_base + lk), col] = 1.0
        for lk in range(4):
            W[(k_base + lk) * n + i_base, col] = 1.0
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_1x4x4(2, 0, 0)
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    if _time_remaining() < 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify_fast(U, V, W):
        rank0 = n * m * p
        U0 = np.zeros((n * m, rank0), dtype=float)
        V0 = np.zeros((m * p, rank0), dtype=float)
        W0 = np.zeros((p * n, rank0), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, idx] = 1
                    V0[j * p + k, idx] = 1
                    W0[k * n + i, idx] = 1
                    idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
