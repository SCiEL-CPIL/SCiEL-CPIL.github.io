import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_SEED_START = time.time()

def _deadline_guard(seconds_remaining=115.0):
    return (time.time() - _SEED_START) > seconds_remaining

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen patterns
    U_patterns = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ])
    V_patterns = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]]
    ])
    W_patterns = np.array([
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ])

    for b in range(8):
        a_block = b // 4
        b_subblock = b % 4
        
        a_col_offset = a_block * 2
        b_row_offset = a_block * 2
        b_col_offset = b_subblock * 2
        
        for r in range(7):
            col = b * 7 + r
            
            for i in range(2):
                for j in range(2):
                    U[i * m + (a_col_offset + j), col] = U_patterns[r, i, j]
            
            for j in range(2):
                for k in range(2):
                    V[(b_row_offset + j) * p + (b_col_offset + k), col] = V_patterns[r, j, k]
            
            for i in range(2):
                for k in range(2):
                    W[(b_col_offset + k) * n + i, col] = W_patterns[r, i, k]

    # Correct Validation
    # Target tensor T_target[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Otherwise 0.
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    # Reconstruct T_recon
    # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # Efficient computation:
    # T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1

    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in W]

    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
