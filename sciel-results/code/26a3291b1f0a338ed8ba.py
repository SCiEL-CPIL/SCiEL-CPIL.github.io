import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix.
    Uses a deterministic Strassen-based decomposition (rank 42).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * group
        global_row = local_i
        return global_row * m + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * group
        global_col = local_j + 2 * block_col
        return global_row * p + global_col

    def get_c_idx(local_i, local_j, block_col):
        global_col = local_j + 2 * block_col
        global_row = local_i
        return global_col * n + global_row

    col = 0
    for group in [0, 1]:
        for block_col in [0, 1, 2]:
            # Term 1: P1 = (A00+A11)(B00+B11)
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 2: P2 = (A10+A11)B00
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1
            
            # Term 3: P3 = A00(B10-B11)
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 4: P4 = A11(B10-B00)
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1
            
            # Term 5: P5 = (A00+A10)B11
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1
            
            # Term 6: P6 = (A10-A00)(B00+B10)
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            # Term 7: P7 = (A01-A11)(B11+B01)
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Validation
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    valid = True
    # Sample a few entries for quick check if time is tight, but 48 checks is fast
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Fallback
    rank_sb = n * m * p
    U_f = [[0] * rank_sb for _ in range(n * m)]
    V_f = [[0] * rank_sb for _ in range(m * p)]
    W_f = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {'U': U_f, 'V': V_f, 'W': W_f}
