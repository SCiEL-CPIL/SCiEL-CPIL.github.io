import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, m, p, n):
    """
    Adds 7 rank-1 terms corresponding to a Strassen decomposition of a 2x2x2 block.
    """
    u_a = a_rows[0] * m + a_cols[0]
    u_b = a_rows[0] * m + a_cols[1]
    u_c = a_rows[1] * m + a_cols[0]
    u_d = a_rows[1] * m + a_cols[1]
    
    v_e = b_rows[0] * p + b_cols[0]
    v_f = b_rows[0] * p + b_cols[1]
    v_g = b_rows[1] * p + b_cols[0]
    v_h = b_rows[1] * p + b_cols[1]
    
    w_00 = c_cols[0] * n + c_rows[0]
    w_01 = c_cols[1] * n + c_rows[0]
    w_10 = c_cols[0] * n + c_rows[1]
    w_11 = c_cols[1] * n + c_rows[1]
    
    # M1
    U[u_a, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M2
    U[u_c, col] += 1.0
    U[u_d, col] += 1.0
    V[v_e, col] += 1.0
    W[w_10, col] += 1.0
    W[w_11, col] += -1.0
    col += 1
    
    # M3
    U[u_a, col] += 1.0
    V[v_f, col] += 1.0
    V[v_h, col] += -1.0
    W[w_01, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M4
    U[u_d, col] += 1.0
    V[v_g, col] += 1.0
    V[v_e, col] += -1.0
    W[w_00, col] += 1.0
    W[w_10, col] += 1.0
    col += 1
    
    # M5
    U[u_a, col] += 1.0
    U[u_b, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += -1.0
    W[w_01, col] += 1.0
    col += 1
    
    # M6
    U[u_c, col] += 1.0
    U[u_a, col] += -1.0
    V[v_e, col] += 1.0
    V[v_f, col] += 1.0
    W[w_11, col] += 1.0
    col += 1
    
    # M7
    U[u_b, col] += 1.0
    U[u_d, col] += -1.0
    V[v_g, col] += 1.0
    V[v_h, col] += 1.0
    W[w_00, col] += 1.0
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Rank calculation:
    # Top part (2x4x8): Split into two 2x2x8 blocks. Each 2x2x8 uses 4 Strassen 2x2x2 blocks (rank 7 each).
    # 2 * (4 * 7) = 56
    # Bottom part (1x4x8): Schoolbook. 1 * 4 * 8 = 32
    # Total Rank = 56 + 32 = 88
    
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :]
    
    # 1a: A[0:2, 0:2] * B[0:2, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
        
    # 1b: A[0:2, 2:4] * B[2:4, :]
    for k_start in range(0, 8, 2):
        col = _strassen_block(
            U, V, W, col,
            a_rows=[0, 1], a_cols=[2, 3],
            b_rows=[2, 3], b_cols=[k_start, k_start+1],
            c_rows=[0, 1], c_cols=[k_start, k_start+1],
            m=m, p=p, n=n
        )
        
    # Part 2: C[2:3, :] = A[2:3, 0:4] * B[0:4, :]
    # Schoolbook for 1x4x8
    for k in range(8):
        for j in range(4):
            U[2 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 2, col] = 1.0
            col += 1
            
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
