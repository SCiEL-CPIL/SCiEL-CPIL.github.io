import numpy as np
import time
import sys

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    # Fallback: schoolbook rank 96
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() - _START_TIME > _DEADLINE:
        return schoolbook()

    # Rank 88 Construction
    # Part 1: i=0, 1x4x8 schoolbook (32 terms)
    # Part 2: i=1,2, 2x4x8 Strassen (56 terms)
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0
    for j in range(m):
        for k in range(p):
            U[j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n, col] = 1.0
            col += 1
            
    # Part 2: i=1,2
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    S_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # 2x4x8 split into 8 blocks of 2x2x2
    # j_blocks: 2 (j=0,1 and j=2,3)
    # k_blocks: 4 (k=0,1; k=2,3; k=4,5; k=6,7)
    
    for j_block in range(2):
        j_base = j_block * 2
        for k_block in range(4):
            k_base = k_block * 2
            for t in range(7):
                # Update U (i=1,2; j=j_base..j_base+1)
                for l in range(4):
                    if S_A[t, l] != 0:
                        loc_i = l // 2
                        loc_j = l % 2
                        g_i = 1 + loc_i
                        g_j = j_base + loc_j
                        U[g_i * m + g_j, col] += S_A[t, l]
                
                # Update V (j=j_base..j_base+1; k=k_base..k_base+1)
                for l in range(4):
                    if S_B[t, l] != 0:
                        loc_j = l // 2
                        loc_k = l % 2
                        g_j = j_base + loc_j
                        g_k = k_base + loc_k
                        V[g_j * p + g_k, col] += S_B[t, l]
                
                # Update W (k=k_base..k_base+1; i=1,2)
                for l in range(4):
                    if S_C[t, l] != 0:
                        loc_i = l // 2
                        loc_k = l % 2
                        g_i = 1 + loc_i
                        g_k = k_base + loc_k
                        W[g_k * n + g_i, col] += S_C[t, l]
                        
                col += 1
                
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
