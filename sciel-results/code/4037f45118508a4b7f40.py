"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 10.0 # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 matrices
    # U[i*2+j, s], V[j*2+k, s], W[k*2+i, s]
    U_sub = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, -1],
        [0, 1, 0, 1, 1, 0, -1]
    ], dtype=int)
    
    V_sub = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=int)
    
    W_sub = np.array([
        [0, -1, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, -1, 0],
        [0, 1, 0, 1, 1, 0, -1]
    ], dtype=int)

    rank_block = 7
    rank_sub = 7
    rank_444 = rank_block * rank_sub
    rank_441 = n * m # 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Fill 4x4x4 part (rank 49)
    for l in range(rank_block):
        for s in range(rank_sub):
            r = l * rank_sub + s
            
            # Update U
            # U[i*4+j, r] = U_sub[i'*2+j', s] * U_sub[I*2+J, l]
            # i = 2I + i', j = 2J + j'
            for I in range(2):
                for J in range(2):
                    u_block_val = U_sub[I*2 + J, l]
                    if u_block_val == 0: continue
                    for i_prime in range(2):
                        for j_prime in range(2):
                            u_sub_val = U_sub[i_prime*2 + j_prime, s]
                            if u_sub_val == 0: continue
                            i = 2*I + i_prime
                            j = 2*J + j_prime
                            U[i*4 + j, r] = u_block_val * u_sub_val
            
            # Update V
            # V[j*4+k, r] = V_sub[j'*2+k', s] * V_sub[J*2+K, l]
            for J in range(2):
                for K in range(2):
                    v_block_val = V_sub[J*2 + K, l]
                    if v_block_val == 0: continue
                    for j_prime in range(2):
                        for k_prime in range(2):
                            v_sub_val = V_sub[j_prime*2 + k_prime, s]
                            if v_sub_val == 0: continue
                            j = 2*J + j_prime
                            k = 2*K + k_prime
                            V[j*4 + k, r] = v_block_val * v_sub_val
                            
            # Update W
            # W[k*4+i, r] = W_sub[k'*2+i', s] * W_sub[K*2+I, l]
            for K in range(2):
                for I in range(2):
                    w_block_val = W_sub[K*2 + I, l]
                    if w_block_val == 0: continue
                    for k_prime in range(2):
                        for i_prime in range(2):
                            w_sub_val = W_sub[k_prime*2 + i_prime, s]
                            if w_sub_val == 0: continue
                            k = 2*K + k_prime
                            i = 2*I + i_prime
                            W[k*4 + i, r] = w_block_val * w_sub_val
                            
    # Fill 4x4x1 part (rank 16)
    # k = 4
    for i in range(n):
        for j in range(m):
            r = rank_444 + i*m + j
            U[i*m + j, r] = 1
            V[j*p + 4, r] = 1
            W[4*n + i, r] = 1
            
    if time.time() > deadline:
         # If we are running out of time, we still return the current result
         pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
