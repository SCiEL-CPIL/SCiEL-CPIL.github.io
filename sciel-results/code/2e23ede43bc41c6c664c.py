"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target rank 44 based on 4 blocks of rank 11
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # We process 4 blocks of 2 columns each
    for s in range(4):
        base_rank = s * 11
        # B columns are 2s and 2s+1
        k0 = 2 * s
        k1 = 2 * s + 1
        
        # --- Strassen Part (7 terms) ---
        
        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        U[0 * m + 0][base_rank + 0] = 1
        U[1 * m + 1][base_rank + 0] = 1
        # V: B00, B11 (B row 0 col k0, B row 1 col k1)
        V[0 * p + k0][base_rank + 0] = 1
        V[1 * p + k1][base_rank + 0] = 1
        # W: C00 += 1, C11 += 1
        W[k0 * n + 0][base_rank + 0] = 1
        W[k1 * n + 1][base_rank + 0] = 1
        
        # Term 2: M2 = (A10 + A11) * B00
        U[1 * m + 0][base_rank + 1] = 1
        U[1 * m + 1][base_rank + 1] = 1
        V[0 * p + k0][base_rank + 1] = 1
        # W: C10 += 1, C11 += -1
        W[k0 * n + 1][base_rank + 1] = 1
        W[k1 * n + 1][base_rank + 1] = -1
        
        # Term 3: M3 = A00 * (B10 - B00)
        U[0 * m + 0][base_rank + 2] = 1
        V[1 * p + k0][base_rank + 2] = 1
        V[0 * p + k0][base_rank + 2] = -1
        # W: C01 += 1, C11 += 1
        W[k1 * n + 0][base_rank + 2] = 1
        W[k1 * n + 1][base_rank + 2] = 1
        
        # Term 4: M4 = A11 * (B01 - B11)
        U[1 * m + 1][base_rank + 3] = 1
        V[0 * p + k1][base_rank + 3] = 1
        V[1 * p + k1][base_rank + 3] = -1
        # W: C00 += 1, C10 += 1
        W[k0 * n + 0][base_rank + 3] = 1
        W[k0 * n + 1][base_rank + 3] = 1
        
        # Term 5: M5 = (A00 + A01) * B11
        U[0 * m + 0][base_rank + 4] = 1
        U[0 * m + 1][base_rank + 4] = 1
        V[1 * p + k1][base_rank + 4] = 1
        # W: C00 += -1, C01 += 1
        W[k0 * n + 0][base_rank + 4] = -1
        W[k1 * n + 0][base_rank + 4] = 1
        
        # Term 6: M6 = (A10 - A00) * (B00 + B10)
        U[1 * m + 0][base_rank + 5] = 1
        U[0 * m + 0][base_rank + 5] = -1
        V[0 * p + k0][base_rank + 5] = 1
        V[1 * p + k0][base_rank + 5] = 1
        # W: C11 += 1
        W[k1 * n + 1][base_rank + 5] = 1
        
        # Term 7: M7 = (A01 - A11) * (B01 + B11)
        U[0 * m + 1][base_rank + 6] = 1
        U[1 * m + 1][base_rank + 6] = -1
        V[0 * p + k1][base_rank + 6] = 1
        V[1 * p + k1][base_rank + 6] = 1
        # W: C00 += 1
        W[k0 * n + 0][base_rank + 6] = 1
        
        # --- Extra Part (4 terms) for A_col2 and B_row2 ---
        
        # Term 8: A02 * B20
        U[0 * m + 2][base_rank + 7] = 1
        V[2 * p + k0][base_rank + 7] = 1
        # W: C00 += 1
        W[k0 * n + 0][base_rank + 7] = 1
        
        # Term 9: A02 * B21
        U[0 * m + 2][base_rank + 8] = 1
        V[2 * p + k1][base_rank + 8] = 1
        # W: C01 += 1
        W[k1 * n + 0][base_rank + 8] = 1
        
        # Term 10: A12 * B20
        U[1 * m + 2][base_rank + 9] = 1
        V[2 * p + k0][base_rank + 9] = 1
        # W: C10 += 1
        W[k0 * n + 1][base_rank + 9] = 1
        
        # Term 11: A12 * B21
        U[1 * m + 2][base_rank + 10] = 1
        V[2 * p + k1][base_rank + 10] = 1
        # W: C11 += 1
        W[k1 * n + 1][base_rank + 10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
