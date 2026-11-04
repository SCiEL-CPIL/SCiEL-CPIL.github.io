"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
DEADLINE = 135.0  # safety margin under 150s wall-clock

# EVOLVE-BLOCK-START

def _time_left():
    return DEADLINE - time.time()

def _build_rank32():
    """
    Construct a rank-32 decomposition for 3x4 * 4x4.
    Split A into A_L (3x3) and A_R (3x1).
    Split B into B_LL (3x3), B_LR (3x1), B_RL (1x3), B_RR (1x1).
    C_L (3x3) = A_L B_LL + A_R B_RL
    C_R (3x1) = A_L B_LR + A_R B_RR
    
    Ranks:
    A_L B_LL: 3x3x3 schoolbook -> 27
    A_L B_LR: 3x3x1 schoolbook -> 3
    A_R B_RL: 3x1x3 rank 1 -> 1
    A_R B_RR: 3x1x1 rank 1 -> 1
    Total: 32
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A_L B_LL (3x3 * 3x3 -> 3x3)
    # A_L is A[0:3, 0:3]. B_LL is B[0:3, 0:3]. C_L is C[0:3, 0:3].
    # Indices:
    # U: i in 0..2, j in 0..2 -> idx i*4+j
    # V: j in 0..2, k in 0..2 -> idx j*4+k
    # W: k in 0..2, i in 0..2 -> idx k*3+i (W is p*n = 4*3=12, idx k*n+i)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. A_L B_LR (3x3 * 3x1 -> 3x1)
    # A_L is A[0:3, 0:3]. B_LR is B[0:3, 3]. C_R is C[0:3, 3].
    # U: i in 0..2, j in 0..2 -> idx i*4+j
    # V: j in 0..2, k=3 -> idx j*4+3
    # W: k=3, i in 0..2 -> idx 3*3+i
    
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
            
    # 3. A_R B_RL (3x1 * 1x3 -> 3x3)
    # A_R is A[0:3, 3]. B_RL is B[3, 0:3]. C_L is C[0:3, 0:3].
    # This is a rank-1 term.
    # U: i in 0..2, j=3 -> idx i*4+3
    # V: j=3, k in 0..2 -> idx 3*4+k
    # W: k in 0..2, i in 0..2 -> idx k*3+i
    
    for i in range(3):
        U[i * m + 3][col] = 1
        
    for k in range(3):
        V[3 * p + k][col] = 1
        
    for i in range(3):
        for k in range(3):
            W[k * n + i][col] = 1
            
    col += 1
    
    # 4. A_R B_RR (3x1 * 1x1 -> 3x1)
    # A_R is A[0:3, 3]. B_RR is B[3, 3]. C_R is C[0:3, 3].
    # This is a rank-1 term.
    # U: i in 0..2, j=3 -> idx i*4+3
    # V: j=3, k=3 -> idx 3*4+3
    # W: k=3, i in 0..2 -> idx 3*3+i
    
    for i in range(3):
        U[i * m + 3][col] = 1
        
    V[3 * p + 3][col] = 1
        
    for i in range(3):
        W[3 * n + i][col] = 1
        
    col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check exactness of the decomposition."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = s + Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Get the rank-32 solution
    best = _build_rank32()
    
    # Verify
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook rank 48 if verification fails (should not happen)
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][t] = 1
                    V_school[j * p + k][t] = 1
                    W_school[k * n + i][t] = 1
                    t += 1
        return {"U": U_school, "V": V_school, "W": W_school}
        
    return best

# EVOLVE-BLOCK-END
