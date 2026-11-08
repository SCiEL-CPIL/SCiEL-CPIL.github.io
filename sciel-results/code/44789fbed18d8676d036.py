import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 2x2x2 decomposition factors
    # For 2x2x2: U is 4x7, V is 4x7, W is 4x7
    # Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Standard Strassen 7-term decomposition
    # M1 = A11(B11+B22)
    # M2 = (A11+A22)B22
    # M3 = (A11+A12)B11
    # M4 = (A21+A22)(B11+B12)
    # M5 = A11B11
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A11-A12)B11
    #
    # C11 = M1 + M4 - M2 + M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M5 + M4 + M6

    # Strassen factors as (local_index, value) pairs
    # local_index = local_i * 2 + local_j for U, local_j * 2 + local_k for V, local_k * 2 + local_i for W
    
    strassen_U = [
        [(0, 1), (3, 1)],   # r=0: A11+A22
        [(2, 1), (3, 1)],   # r=1: A21+A22
        [(0, 1)],           # r=2: A11
        [(3, 1)],           # r=3: A22
        [(0, 1), (1, 1)],   # r=4: A11+A12
        [(2, 1), (0, -1)],  # r=5: A21-A11
        [(1, 1), (3, -1)],  # r=6: A12-A22
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # r=0: B11+B22
        [(0, 1)],           # r=1: B11
        [(1, 1), (3, -1)],  # r=2: B21-B22
        [(2, 1), (0, -1)],  # r=3: B12-B11
        [(3, 1)],           # r=4: B22
        [(0, 1), (1, 1)],   # r=5: B11+B12
        [(2, 1), (3, 1)],   # r=6: B21+B22
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # r=0: C11+C22
        [(1, 1), (3, -1)],  # r=1: C12-C22
        [(2, 1), (3, 1)],   # r=2: C21+C22
        [(0, 1), (1, 1)],   # r=3: C11+C12
        [(0, -1), (2, 1)],  # r=4: -C11+C21
        [(3, 1)],           # r=5: C22
        [(0, 1)],           # r=6: C11
    ]

    # Apply 8 blocks of 2x2x2 Strassen
    # Block b: i_off=0, j_off=2*(b//4), k_off=2*(b%4)
    for b in range(8):
        if time.time() > deadline:
            break
            
        i_off = 0
        j_off = 2 * (b // 4)
        k_off = 2 * (b % 4)
        base = 7 * b
        
        for r in range(7):
            cr = base + r
            
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                gi = local_i + i_off
                gj = local_j + j_off
                U[gi * 4 + gj, cr] += val

            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                gj = local_j + j_off
                gk = local_k + k_off
                V[gj * 8 + gk, cr] += val

            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                gk = local_k + k_off
                gi = local_i + i_off
                W[gk * 2 + gi, cr] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
