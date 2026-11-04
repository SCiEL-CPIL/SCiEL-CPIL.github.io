import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Build the multiplication tensor for verification
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Rank-42 Strassen-based decomposition
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((n*p, rank))
    
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(6):
        s = b // 3
        k_block = b % 3
        for t in range(7):
            c = 7 * b + t
            u_coeff = strassen_A[t]
            for i in range(2):
                for j_loc in range(2):
                    idx_A = i * 2 + j_loc
                    coeff = u_coeff[idx_A]
                    if coeff != 0:
                        global_i = i
                        global_j = 2 * s + j_loc
                        U[global_i * m + global_j][c] = coeff
            v_coeff = strassen_B[t]
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_B = j_loc * 2 + k_loc
                    coeff = v_coeff[idx_B]
                    if coeff != 0:
                        global_j = 2 * s + j_loc
                        global_k = 2 * k_block + k_loc
                        V[global_j * p + global_k][c] = coeff
            w_coeff = w_coeffs[t]
            for i in range(2):
                for k_loc in range(2):
                    idx_W = i * 2 + k_loc
                    coeff = w_coeff[idx_W]
                    if coeff != 0:
                        global_k = 2 * k_block + k_loc
                        global_i = i
                        W[global_k * n + global_i][c] = coeff
    
    # Verify
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    if not np.allclose(T_recon, T):
        # Fallback to naive
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((n*p, rank))
        for t in range(rank):
            i = t // (m*p)
            rem = t % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j, t] = 1
            V[j*p+k, t] = 1
            W[k*n+i, t] = 1
    
    U_list = [[U[i][t] for t in range(rank)] for i in range(n*m)]
    V_list = [[V[i][t] for t in range(rank)] for i in range(m*p)]
    W_list = [[W[i][t] for t in range(rank)] for i in range(n*p)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
