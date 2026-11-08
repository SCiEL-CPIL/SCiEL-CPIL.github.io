"""Rewritten program for matrix multiplication 3,4,8 using rank-80/88 decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    
    def get_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a,b,c] = 1.0
        return T

    def try_find_decomposition(n, m, p, target_rank, seed=42, timeout=5.0):
        T = get_tensor(n, m, p)
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        start_time = time.time()
        
        for step in range(2000):
            if time.time() - start_time > timeout:
                break
            
            T_curr = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_curr
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            lr = 0.05
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if np.linalg.norm(diff) < 1e-6:
                break
        
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        if np.allclose(T, T_check, atol=1e-9):
            return U_int, V_int, W_int
        return None

    decomp_24 = try_find_decomposition(2, 4, 4, 24)
    
    if decomp_24 is not None:
        U24, V24, W24 = decomp_24
        total_rank = 32 + 2 * 24
    else:
        total_rank = 88

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    if decomp_24 is not None:
        for block_idx in range(2):
            k_start = block_idx * 4
            for r in range(24):
                for i_loc in range(2):
                    for j in range(4):
                        val = U24[i_loc * 4 + j, r]
                        if val != 0:
                            global_i = 1 + i_loc
                            U[global_i * m + j][col] += val
                for j in range(4):
                    for k_loc in range(4):
                        val = V24[j * 4 + k_loc, r]
                        if val != 0:
                            global_k = k_start + k_loc
                            V[j * p + global_k][col] += val
                for k_loc in range(4):
                    for i_loc in range(2):
                        val = W24[k_loc * 2 + i_loc, r]
                        if val != 0:
                            global_k = k_start + k_loc
                            global_i = 1 + i_loc
                            W[global_k * n + global_i][col] += val
                col += 1
    else:
        strassen_U = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
        strassen_V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
        strassen_W = [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                for t in range(7):
                    for l in range(4):
                        if strassen_U[t][l] != 0:
                            local_i_prime = l // 2
                            local_j_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_j = j_start + local_j_prime
                            U[global_i * m + global_j][col] += strassen_U[t][l]
                    for l in range(4):
                        if strassen_V[t][l] != 0:
                            local_j_prime = l // 2
                            local_k_prime = l % 2
                            global_j = j_start + local_j_prime
                            global_k = k_start + local_k_prime
                            V[global_j * p + global_k][col] += strassen_V[t][l]
                    for l in range(4):
                        if strassen_W[t][l] != 0:
                            local_i_prime = l // 2
                            local_k_prime = l % 2
                            global_i = 1 + local_i_prime
                            global_k = k_start + local_k_prime
                            W[global_k * n + global_i][col] += strassen_W[t][l]
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
