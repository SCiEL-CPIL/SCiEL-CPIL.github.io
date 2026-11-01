"""Program for matrix multiplication 2,2,13 - seeking rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 13]
DEADLINE = time.time() + 140.0  # Safety margin before 150s kill

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Uses a structured decomposition:
    - 6 Strassen blocks (rank 7 each) for column pairs (0,1), (2,3), ..., (10,11)
    - 1 direct block (rank 4) for the last column (12)
    Total rank: 6*7 + 4 = 46.
    """
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    # Strassen coefficients for A (2x2)
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # Indices for U (i*m + j)
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # For each of the 6 blocks
    for t in range(6):
        k0 = 2 * t
        k1 = 2 * t + 1
        base_rank = 7 * t
        
        # Strassen coefficients for B (2x2 block)
        # M1: B0k0 + B1k1
        # M2: B0k0
        # M3: B1k0 - B0k0
        # M4: B0k1 - B1k1
        # M5: B1k1
        # M6: B0k0 + B0k1
        # M7: B1k0 + B1k1
        
        # V indices: j*p + k
        # B0k0: 0*p + k0 = k0
        # B0k1: 0*p + k1 = k1
        # B1k0: 1*p + k0 = p + k0
        # B1k1: 1*p + k1 = p + k1
        
        V_strassen = [
            [0]*p + [0]*p, # placeholder, will fill
            [0]*p + [0]*p,
            [0]*p + [0]*p,
            [0]*p + [0]*p,
            [0]*p + [0]*p,
            [0]*p + [0]*p,
            [0]*p + [0]*p
        ]
        
        # Fill V for this block
        # M1
        V_strassen[0][k0] = 1
        V_strassen[0][p + k1] = 1
        # M2
        V_strassen[1][k0] = 1
        # M3
        V_strassen[2][p + k0] = 1
        V_strassen[2][k0] = -1
        # M4
        V_strassen[3][k1] = 1
        V_strassen[3][p + k1] = -1
        # M5
        V_strassen[4][p + k1] = 1
        # M6
        V_strassen[5][k0] = 1
        V_strassen[5][k1] = 1
        # M7
        V_strassen[6][p + k0] = 1
        V_strassen[6][p + k1] = 1
        
        # W coefficients for C
        # C0k0 = M1 + M4 - M5 + M7
        # C0k1 = M3 + M5
        # C1k0 = M2 + M4
        # C1k1 = M1 - M2 + M3 + M6
        
        # W indices: k*n + i
        # C0k0: k0*2 + 0 = 2*k0
        # C0k1: k1*2 + 0 = 2*k1
        # C1k0: k0*2 + 1 = 2*k0 + 1
        # C1k1: k1*2 + 1 = 2*k1 + 1
        
        # We need to add contributions to W for each rank term
        # W[r] is a vector of length p*n
        # For term r in this block, global index is base_rank + r
        
        # M1 (r=0)
        W[2*k0][base_rank + 0] += 1
        W[2*k1 + 1][base_rank + 0] += 1
        # M2 (r=1)
        W[2*k0 + 1][base_rank + 1] += 1
        W[2*k1 + 1][base_rank + 1] += -1
        # M3 (r=2)
        W[2*k1][base_rank + 2] += 1
        W[2*k1 + 1][base_rank + 2] += 1
        # M4 (r=3)
        W[2*k0][base_rank + 3] += 1
        W[2*k0 + 1][base_rank + 3] += 1
        # M5 (r=4)
        W[2*k0][base_rank + 4] += -1
        W[2*k1][base_rank + 4] += 1
        # M6 (r=5)
        W[2*k1 + 1][base_rank + 5] += 1
        # M7 (r=6)
        W[2*k0][base_rank + 6] += 1
        
        # Fill U and V for this block
        for r in range(7):
            global_r = base_rank + r
            for i in range(n):
                for j in range(m):
                    U[i*m + j][global_r] = U_strassen[r][i*m + j]
            for j in range(m):
                for k in range(p):
                    V[j*p + k][global_r] = V_strassen[r][j*p + k]
                    
    # Last column k=12, direct terms (rank 4)
    k_last = 12
    base_rank = 42
    
    # Terms:
    # 1: A00 * B0k -> C0k
    # 2: A01 * B1k -> C0k
    # 3: A10 * B0k -> C1k
    # 4: A11 * B1k -> C1k
    
    # U
    U[0][base_rank + 0] = 1 # A00
    U[1][base_rank + 1] = 1 # A01
    U[2][base_rank + 2] = 1 # A10
    U[3][base_rank + 3] = 1 # A11
    
    # V
    V[0*p + k_last][base_rank + 0] = 1 # B0k
    V[1*p + k_last][base_rank + 1] = 1 # B1k
    V[0*p + k_last][base_rank + 2] = 1 # B0k
    V[1*p + k_last][base_rank + 3] = 1 # B1k
    
    # W
    # C0k index: k*2 + 0
    # C1k index: k*2 + 1
    W[k_last*n + 0][base_rank + 0] = 1
    W[k_last*n + 0][base_rank + 1] = 1
    W[k_last*n + 1][base_rank + 2] = 1
    W[k_last*n + 1][base_rank + 3] = 1
    
    return {"U": U, "V": V, "W": W}
