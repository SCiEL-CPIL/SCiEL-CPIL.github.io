import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 145.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    for j1 in range(2):
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                col += 1
        
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j][col] = 1
                global_j = 2 * j1 + j2
                global_k = k
                V[global_j * p + global_k][col] = 1
                global_i = 2
                global_k = k
                W[global_k * n + global_i][col] = 1
                col += 1

    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_time=100.0):
        rank = U_mat.shape[1]
        
        U_idx = np.zeros((n, m, rank))
        for i in range(n):
            for j in range(m):
                U_idx[i, j, :] = U_mat[i*m+j, :]
        
        V_idx = np.zeros((n, m, p, rank))
        for j in range(m):
            for k in range(p):
                V_idx[:, j, k, :] = V_mat[j*p+k, :]
        
        W_idx = np.zeros((n, m, p, rank))
        for k in range(p):
            for i in range(n):
                W_idx[i, :, k, :] = W_mat[k*n+i, :]
        
        T = U_idx[:, :, np.newaxis, :] * V_idx * W_idx
        total = T.sum(axis=3)
        
        if not np.allclose(total, 1.0, atol=1e-9):
            return U_mat, V_mat, W_mat
        
        keep = [True] * rank
        start_time = time.time()
        changed = True
        while changed and (time.time() - start_time < max_time):
            changed = False
            for r in range(rank):
                if time.time() - start_time > max_time:
                    break
                if not keep[r]:
                    continue
                
                new_total = total - T[:, :, :, r]
                if np.allclose(new_total, 1.0, atol=1e-9):
                    keep[r] = False
                    changed = True
        
        idx = [i for i in range(rank) if keep[i]]
        if len(idx) < rank:
            return U_mat[:, idx], V_mat[:, idx], W_mat[:, idx]
        
        return U_mat, V_mat, W_mat
    
    U_opt, V_opt, W_opt = try_reduce_rank(U_np, V_np, W_np, max_time=100.0)
    
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U_opt), "V": to_list(V_opt), "W": to_list(W_opt)}
# EVOLVE-BLOCK-END
