"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 150.0

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    outer_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    outer_C = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank114():
    """
    Build rank-114 decomposition: 2 * M_{4,4,4} + M_{4,4,1}.
    M_{4,4,4} rank 49.
    M_{4,4,1} rank 16.
    Total 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    
    U4, V4, W4 = _decomp_4x4x4()
    r4 = U4.shape[1] # 49
    
    U1, V1, W1 = _schoolbook_nm_p(4, 4, 1)
    r1 = U1.shape[1] # 16
    
    rank = r4 + r4 + r1
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: Columns 0-3 of B (4x4)
    col_offset = 0
    U[:, col_offset:col_offset+r4] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + k, col_offset:col_offset+r4] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, col_offset:col_offset+r4] = W4[k * n + i, :]
    col_offset += r4
    
    # Block 2: Columns 4-7 of B (4x4)
    U[:, col_offset:col_offset+r4] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), col_offset:col_offset+r4] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, col_offset:col_offset+r4] = W4[k * n + i, :]
    col_offset += r4
    
    # Block 3: Column 8 of B (4x1)
    U[:, col_offset:col_offset+r1] = U1
    for j in range(m):
        V[j * p + 8, col_offset:col_offset+r1] = V1[j * 1 + 0, :]
    for i in range(n):
        W[8 * n + i, col_offset:col_offset+r1] = W1[0 * n + i, :]
        
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()
    
    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]
    
    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)
    
    return np.all(sums == 1)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_result = None
    best_rank = n * m * p  # schoolbook fallback
    
    # Try rank-114 decomposition
    try:
        U114, V114, W114 = _build_rank114()
        if _verify_fast(U114, V114, W114, n, m, p):
            best_result = (U114, V114, W114)
            best_rank = U114.shape[1]
    except Exception:
        pass
    
    # Deadline guard
    if time.time() - start_time > DEADLINE - 5:
        if best_result is not None:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
