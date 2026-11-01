import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 14 decomposition
    rank = 14
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Reconstruction:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We use this for the first two columns (0,1) and the next two (2,3).
    
    def set_strassen_block(offset_v, offset_w, base_r):
        for r_local in range(7):
            r = base_r + r_local
            
            # U coefficients (same for all blocks)
            # Index in U: i*m + j = 2*i + j
            if r_local == 0: # M1: a00+a11
                U[0][r] = 1.0
                U[3][r] = 1.0
            elif r_local == 1: # M2: a10+a11
                U[2][r] = 1.0
                U[3][r] = 1.0
            elif r_local == 2: # M3: a00
                U[0][r] = 1.0
            elif r_local == 3: # M4: a11
                U[3][r] = 1.0
            elif r_local == 4: # M5: a00+a01
                U[0][r] = 1.0
                U[1][r] = 1.0
            elif r_local == 5: # M6: a10-a00
                U[2][r] = 1.0
                U[0][r] = -1.0
            elif r_local == 6: # M7: a01-a11
                U[1][r] = 1.0
                U[3][r] = -1.0
            
            # V coefficients
            # Index in V: j*p + k = 4*j + k
            # j is row of B (0,1), k is col of B (offset_v, offset_v+1)
            if r_local == 0: # b00+b11 -> b(j=0, k=offset_v) + b(j=1, k=offset_v+1)
                V[offset_v][r] = 1.0
                V[4 + offset_v + 1][r] = 1.0
            elif r_local == 1: # b00 -> b(j=0, k=offset_v)
                V[offset_v][r] = 1.0
            elif r_local == 2: # b01-b11 -> b(j=0, k=offset_v+1) - b(j=1, k=offset_v+1)
                V[offset_v + 1][r] = 1.0
                V[4 + offset_v + 1][r] = -1.0
            elif r_local == 3: # b10-b00 -> b(j=1, k=offset_v) - b(j=0, k=offset_v)
                V[4 + offset_v][r] = 1.0
                V[offset_v][r] = -1.0
            elif r_local == 4: # b11 -> b(j=1, k=offset_v+1)
                V[4 + offset_v + 1][r] = 1.0
            elif r_local == 5: # b00+b01 -> b(j=0, k=offset_v) + b(j=0, k=offset_v+1)
                V[offset_v][r] = 1.0
                V[offset_v + 1][r] = 1.0
            elif r_local == 6: # b10+b11 -> b(j=1, k=offset_v) + b(j=1, k=offset_v+1)
                V[4 + offset_v][r] = 1.0
                V[4 + offset_v + 1][r] = 1.0

            # W coefficients
            # Index in W: k*n + i = 2*k + i
            # k is col of C (offset_w, offset_w+1), i is row of C (0,1)
            # C00 (i=0, k=offset_w) = M1 - M4 + M5 - M7
            # C01 (i=0, k=offset_w+1) = M3 + M5
            # C10 (i=1, k=offset_w) = M2 + M4
            # C11 (i=1, k=offset_w+1) = M1 - M2 + M3 + M6
            
            w00 = 2 * offset_w + 0
            w01 = 2 * (offset_w + 1) + 0
            w10 = 2 * offset_w + 1
            w11 = 2 * (offset_w + 1) + 1
            
            if r_local == 0: # M1
                W[w00][r] = 1.0
                W[w11][r] = 1.0
            elif r_local == 1: # M2
                W[w10][r] = 1.0
                W[w11][r] = -1.0
            elif r_local == 2: # M3
                W[w01][r] = 1.0
                W[w11][r] = 1.0
            elif r_local == 3: # M4
                W[w00][r] = -1.0
                W[w10][r] = 1.0
            elif r_local == 4: # M5
                W[w00][r] = 1.0
                W[w01][r] = 1.0
            elif r_local == 5: # M6
                W[w11][r] = 1.0
            elif r_local == 6: # M7
                W[w00][r] = -1.0

    set_strassen_block(0, 0, 0)
    set_strassen_block(2, 2, 7)

    return {"U": U, "V": V, "W": W}
