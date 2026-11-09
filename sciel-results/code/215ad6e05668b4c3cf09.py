import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = build_decomposition(n, m, p)
    
    if not verify(U, V, W, n, m, p):
        return schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    tensor = np.ones((n, m, p), dtype=float)
    
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T
    
    recon = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_vec = U_mat[c].reshape(n, m)
        v_vec = V_mat[c].reshape(m, p)
        w_vec = W_mat[c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    return np.allclose(recon, tensor, atol=1e-10)

def build_decomposition(n, m, p):
    U = [[0] * 57 for _ in range(n * m)]
    V = [[0] * 57 for _ in range(m * p)]
    W = [[0] * 57 for _ in range(p * n)]
    
    _build_strassen_4x4x4(U, V, W, 0, n, m, p)
    
    rank_offset = 49
    for k in [4, 5]:
        for i in range(n):
            for j in range(m):
                U[i * m + j][rank_offset] = 1
                V[j * p + k][rank_offset] = 1
                W[k * n + i][rank_offset] = 1
                rank_offset += 1
    
    return U, V, W

def _build_strassen_4x4x4(U, V, W, rank_start, n, m, p):
    inner_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): 1}),
        ({(0,0): 1, (1,0): 1}, {(1,1): 1, (1,0): -1}),
        ({(1,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
    ]
    
    z_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: -1, 1: 1}
    ]
    
    outer_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): 1}),
        ({(0,0): 1, (1,0): 1}, {(1,1): 1, (1,0): -1}),
        ({(1,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
    ]
    
    outer_reconstruction = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ]
    
    rank = rank_start
    
    for outer_idx in range(7):
        a_blocks, b_blocks = outer_strassen[outer_idx]
        
        for inner_idx in range(7):
            x_coeffs, y_coeffs = inner_strassen[inner_idx]
            
            u_vec = {}
            v_vec = {}
            w_vec = {}
            
            for (lr, lc), x_coeff in x_coeffs.items():
                for (br, bc), a_coeff in a_blocks.items():
                    global_i = br * 2 + lr
                    global_j = bc * 2 + lc
                    coeff = x_coeff * a_coeff
                    key = (global_i, global_j)
                    u_vec[key] = u_vec.get(key, 0) + coeff
            
            for (lr, lc), y_coeff in y_coeffs.items():
                for (br, bc), b_coeff in b_blocks.items():
                    global_j = br * 2 + lr
                    global_k = bc * 2 + lc
                    coeff = y_coeff * b_coeff
                    key = (global_j, global_k)
                    v_vec[key] = v_vec.get(key, 0) + coeff
            
            for (z_idx, z_coeff) in z_contributions[inner_idx].items():
                z_row = z_idx // 2
                z_col = z_idx % 2
                for (c_br, c_bc), c_coeff in outer_reconstruction[outer_idx].items():
                    global_i = c_br * 2 + z_row
                    global_k = c_bc * 2 + z_col
                    total_coeff = z_coeff * c_coeff
                    key = (global_k, global_i)
                    w_vec[key] = w_vec.get(key, 0) + total_coeff
            
            for (i, j), coeff in u_vec.items():
                if coeff != 0:
                    U[i * m + j][rank] = coeff
            for (j, k), coeff in v_vec.items():
                if coeff != 0:
                    V[j * p + k][rank] = coeff
            for (k, i), coeff in w_vec.items():
                if coeff != 0:
                    W[k * n + i][rank] = coeff
            
            rank += 1
    
    return rank - rank_start
