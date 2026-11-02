"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_sb = n * m * p
    
    # Target rank 68 using Strassen for 2x2 blocks
    target_rank = 68
    
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)
    
    rank = 0
    
    # Strassen part: A[0:2, 0:2] * B[0:2, :]
    # B is split into 4 blocks of 2x2
    for block in range(4):
        # Block columns: 2*block, 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen terms
        # M1
        U[0, rank] = 1; U[4, rank] = 1
        V[k1, rank] = 1; V[8 + k1, rank] = 1
        W[k0 * 3 + 0, rank] = 1; W[k1 * 3 + 1, rank] = 1
        rank += 1
        
        # M2
        U[3, rank] = 1; U[4, rank] = 1
        V[k0, rank] = 1
        W[k0 * 3 + 1, rank] = 1; W[k1 * 3 + 1, rank] = -1
        rank += 1
        
        # M3
        U[0, rank] = 1
        V[8 + k0, rank] = 1; V[8 + k1, rank] = -1
        W[k1 * 3 + 0, rank] = 1; W[k1 * 3 + 1, rank] = 1
        rank += 1
        
        # M4
        U[4, rank] = 1
        V[k0, rank] = 1; V[k1, rank] = 1
        W[k0 * 3 + 0, rank] = 1; W[k0 * 3 + 1, rank] = 1
        rank += 1
        
        # M5
        U[0, rank] = 1; U[1, rank] = 1
        V[k0, rank] = 1; V[8 + k0, rank] = -1
        W[k0 * 3 + 0, rank] = -1; W[k1 * 3 + 0, rank] = 1
        rank += 1
        
        # M6
        U[4, rank] = 1; U[0, rank] = -1
        V[k0, rank] = 1; V[k1, rank] = 1
        W[k1 * 3 + 1, rank] = 1
        rank += 1
        
        # M7
        U[0, rank] = 1; U[4, rank] = -1
        V[8 + k0, rank] = 1; V[8 + k1, rank] = 1
        W[k0 * 3 + 0, rank] = 1; W[k1 * 3 + 1, rank] = -1
        rank += 1

    # Schoolbook part 1: A[0:2, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2, rank] = 1
            V[2 * 8 + k, rank] = 1
            W[k * 3 + i, rank] = 1
            rank += 1
            
    # Schoolbook part 2: A[2, :] * B[:, :]
    for j in range(3):
        for k in range(8):
            U[2 * 3 + j, rank] = 1
            V[j * 8 + k, rank] = 1
            W[k * 3 + 2, rank] = 1
            rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
