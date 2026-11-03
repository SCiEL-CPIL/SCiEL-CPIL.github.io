import time
import sys

# Constants for the problem dimensions
N_ROWS_A = 2
M_COLS_A = 4
P_COLS_B = 4

# The known minimal rank for 2x4x4 matrix multiplication is 28.
TARGET_RANK = 28

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    
    # Initialize factor matrices with zeros
    # U: shape (n*m, rank) -> (8, 28)
    # V: shape (m*p, rank) -> (16, 28)
    # W: shape (p*n, rank) -> (8, 28)
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
        (0, 2, 2, 2, 2)
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        if time.time() > deadline:
            break
            
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
        
        # Term 1
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    if current_rank != TARGET_RANK:
        raise ValueError(f"Rank mismatch: expected {TARGET_RANK}, got {current_rank}")
        
    return {"U": U, "V": V, "W": W}
