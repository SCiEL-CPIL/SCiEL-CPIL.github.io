import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    R_total = 57
    U = np.zeros((16, R_total), dtype=int)
    V = np.zeros((24, R_total), dtype=int)
    W = np.zeros((24, R_total), dtype=int)
    
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    
    # Part 1: M_{4,4,4} using two-level Strassen (rank 49)
    for s_outer in range(7):
        for s_inner in range(7):
            for a in range(4):
                if alpha[s_inner][a] != 0:
                    a_row, a_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s_outer][r] != 0:
                            r_block, r_inner = r // 2, r % 2
                            u_idx = (r_block * 2 + a_row) * 4 + (r_inner * 2 + a_col)
                            U[u_idx, col] += alpha[s_outer][r] * alpha[s_inner][a]
            
            for b in range(4):
                if beta[s_inner][b] != 0:
                    b_row, b_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s_outer][c] != 0:
                            c_block, c_inner = c // 2, c % 2
                            v_idx = (c_block * 2 + b_row) * 6 + (c_inner * 2 + b_col)
                            V[v_idx, col] += beta[s_outer][c] * beta[s_inner][b]
            
            for a, b, gamma_outer in recon[s_outer]:
                for za, zb, gamma_inner in recon[s_inner]:
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma_outer * gamma_inner
            
            col += 1
    
    # Part 2: M_{4,4,2} using schoolbook (rank 8)
    for i in range(4):
        for j in range(2):
            for k in range(4):
                U[i * 4 + k, col] += 1
                V[k * 6 + (4 + j), col] += 1
                W[(4 + j) * 4 + i, col] += 1
                col += 1
    
    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
