import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    W_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    col = 0
    num_blocks = p // 2
    
    for t in range(num_blocks):
        for r in range(7):
            U[:, col] = U_s[r]
            
            v00 = 0 * p + 2 * t
            v01 = 0 * p + (2 * t + 1)
            v10 = 1 * p + 2 * t
            v11 = 1 * p + (2 * t + 1)
            
            V[v00, col] = V_s[r][0]
            V[v01, col] = V_s[r][1]
            V[v10, col] = V_s[r][2]
            V[v11, col] = V_s[r][3]
            
            w00 = (2 * t) * n + 0
            w01 = (2 * t + 1) * n + 0
            w10 = (2 * t) * n + 1
            w11 = (2 * t + 1) * n + 1
            
            W[w00, col] = W_s[r][0]
            W[w01, col] = W_s[r][1]
            W[w10, col] = W_s[r][2]
            W[w11, col] = W_s[r][3]
            
            col += 1
    
    # Check for linear dependencies among the 28 rank-1 terms
    if time.time() < deadline - 10:
        tensor_dim = (n * m) * (m * p) * (p * n)
        tensor_matrix = np.zeros((tensor_dim, rank), dtype=float)
        
        for rr in range(rank):
            u_vec = U[:, rr].astype(float)
            v_vec = V[:, rr].astype(float)
            w_vec = W[:, rr].astype(float)
            outer = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
            tensor_matrix[:, rr] = outer.flatten()
        
        tensor_rank = np.linalg.matrix_rank(tensor_matrix)
        # If tensor_rank < rank, there are linear dependencies that might
        # allow a lower-rank decomposition, but we don't have an efficient
        # method to exploit them within the time budget.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
