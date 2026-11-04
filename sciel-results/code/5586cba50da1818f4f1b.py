import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # A is 2x4, B is 4x8, C is 2x8
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    # W indexing: W[k*n + i, r]
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Partition m=4 into 2 groups of 2 (s = 0, 1)
    # Partition p=8 into 4 groups of 2 (j = 0, 1, 2, 3)
    # Each sub-problem: A_s (2x2) * B_{s,j} (2x2) -> C_{s,j} (2x2)
    # 8 sub-problems, each with rank 7 (Strassen)
    
    for s in range(2):
        for j in range(4):
            if time.time() > deadline:
                break
            
            # Global indices for A_s (rows 0,1; cols 2s, 2s+1)
            # A index: i * m + j_global
            a00 = 0 * m + (2 * s + 0)  # = 2s
            a01 = 0 * m + (2 * s + 1)  # = 2s+1
            a10 = 1 * m + (2 * s + 0)  # = 4+2s
            a11 = 1 * m + (2 * s + 1)  # = 4+2s+1
            
            # Global indices for B_{s,j} (rows 2s, 2s+1; cols 2j, 2j+1)
            # B index: j_global * p + k_global
            b00 = (2 * s + 0) * p + (2 * j + 0)
            b01 = (2 * s + 0) * p + (2 * j + 1)
            b10 = (2 * s + 1) * p + (2 * j + 0)
            b11 = (2 * s + 1) * p + (2 * j + 1)
            
            # Global indices for C (rows 0,1; cols 2j, 2j+1)
            # C index: k_global * n + i
            c00 = (2 * j + 0) * n + 0  # = 2j
            c01 = (2 * j + 1) * n + 0  # = 2j+1
            c10 = (2 * j + 0) * n + 1  # = 2j+2
            c11 = (2 * j + 1) * n + 1  # = 2j+3
            
            # Strassen 7 multiplications:
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)b00
            # M3 = a00(b01-b11)
            # M4 = a11(b10-b00)
            # M5 = (a00+a01)b11
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a01-a11)(b10+b11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M1
            U[a00][col] += 1
            U[a11][col] += 1
            V[b00][col] += 1
            V[b11][col] += 1
            W[c00][col] += 1
            W[c11][col] += 1
            
            # M2
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            V[b00][col+1] += 1
            W[c10][col+1] += 1
            W[c11][col+1] += -1
            
            # M3
            U[a00][col+2] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            W[c01][col+2] += 1
            W[c11][col+2] += 1
            
            # M4
            U[a11][col+3] += 1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            W[c00][col+3] += 1
            W[c10][col+3] += 1
            
            # M5
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            V[b11][col+4] += 1
            W[c00][col+4] += -1
            W[c01][col+4] += 1
            
            # M6
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            W[c11][col+5] += 1
            
            # M7
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            W[c00][col+6] += 1
            
            col += 7
    
    return {"U": U, "V": V, "W": W}
