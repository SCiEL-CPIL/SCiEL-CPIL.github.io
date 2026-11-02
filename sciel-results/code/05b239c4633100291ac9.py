"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    Returns exact integer entries (all coefficients are ±1 or 0).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2 (all entries are integers: 0, 1, or -1)
    # U: 7 x 4 (intermediate x A-entry [A00, A01, A10, A11])
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    # V: 7 x 4 (intermediate x B-entry [B00, B01, B10, B11])
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    # W: 7 x 4 (intermediate x C-entry [C00, C01, C10, C11])
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, -1],  # M2: C10+1, C11-1
        [0, 1, 0, 1],   # M3: C01+1, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C10+1
        [-1, 1, 0, 0],  # M5: C00-1, C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0]    # M7: C00+1
    ]
    
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * 7  # 56
    
    # Build the full factor matrices as lists of integers
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7
        
        for s in range(7):
            col = base + s
            
            # U: same for all blocks
            for idx in range(4):
                U[idx][col] = strassen_U[s][idx]
            
            # V: map local B indices to global V indices
            # V row index: j*p + k where j is row of B, k is column
            V[k0][col] = strassen_V[s][0]       # B[0,k0]
            V[k1][col] = strassen_V[s][1]       # B[0,k1]
            V[p + k0][col] = strassen_V[s][2]   # B[1,k0]
            V[p + k1][col] = strassen_V[s][3]   # B[1,k1]
            
            # W: map local C indices to global W indices
            # W row index: k*n + i where k is column of C, i is row of C
            W[2*k0][col] = strassen_W[s][0]     # C[0,k0]
            W[2*k1][col] = strassen_W[s][1]     # C[0,k1]
            W[2*k0+1][col] = strassen_W[s][2]   # C[1,k0]
            W[2*k1+1][col] = strassen_W[s][3]   # C[1,k1]
    
    # Exact verification using integer arithmetic
    # Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    max_err = max(max_err, abs(total - 1))
    
    if max_err > 0:
        # Fall back to schoolbook (rank 64)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
