"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=6
    # Rank 21 decomposition using 3 Strassen blocks
    
    # Strassen coefficients for 2x2x2
    # U_local: 4x7
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    # V_local: 4x7
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    # W_local: 4x7
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    num_blocks = p // 2  # 3 blocks
    
    for k in range(num_blocks):
        base = k * 7
        # U: same for all blocks
        for i in range(4):
            for r in range(7):
                U[i, base + r] = U_local[i][r]
        
        # V: map local B indices to global
        # Local B00 -> Global V[0*6 + 2k] = V[2k]
        # Local B01 -> Global V[0*6 + 2k+1] = V[2k+1]
        # Local B10 -> Global V[1*6 + 2k] = V[6+2k]
        # Local B11 -> Global V[1*6 + 2k+1] = V[7+2k]
        
        for r in range(7):
            V[2*k, base + r] = V_local[0][r]
            V[2*k+1, base + r] = V_local[1][r]
            V[6+2*k, base + r] = V_local[2][r]
            V[7+2*k, base + r] = V_local[3][r]
            
        # W: map local C indices to global
        # Local C00 (i=0, j=0) -> Global W[(2k+0)*2 + 0] = W[4k]
        # Local C01 (i=0, j=1) -> Global W[(2k+1)*2 + 0] = W[4k+2]
        # Local C10 (i=1, j=0) -> Global W[(2k+0)*2 + 1] = W[4k+1]
        # Local C11 (i=1, j=1) -> Global W[(2k+1)*2 + 1] = W[4k+3]
        
        for r in range(7):
            W[4*k, base + r] = W_local[0][r]
            W[4*k+2, base + r] = W_local[1][r]
            W[4*k+1, base + r] = W_local[2][r]
            W[4*k+3, base + r] = W_local[3][r]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
