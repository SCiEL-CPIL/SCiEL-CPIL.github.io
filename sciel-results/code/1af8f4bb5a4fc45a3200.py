import numpy as np
import time

def construct():
    _deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 2, 2, 16
    num_blocks = p // m  # 8 blocks of width 2
    
    # Strassen factors for 2x2x2 matrix multiplication
    # Each factor: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, coeff) for A[i,j], row in U = i*m+j
    # v_entries: list of (j, k, coeff) for B[j,k], row in V = j*p+k
    # w_entries: list of (i, k, coeff) for C[i,k], row in W = k*n+i
    
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    rank = 7 * num_blocks  # 56
    U = np.zeros((n * m, rank))  # 4 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 32 x 56
    
    for t in range(num_blocks):
        k_offset = t * m  # offset in the k (column of B / row of C) dimension
        for s in range(7):
            col = t * 7 + s
            u_entries, v_entries, w_entries = strassen_factors[s]
            
            for (i, j, coeff) in u_entries:
                U[i * m + j, col] = coeff
            
            for (j, k_local, coeff) in v_entries:
                V[j * p + (k_offset + k_local), col] = coeff
            
            for (i, k_local, coeff) in w_entries:
                W[(k_offset + k_local) * n + i, col] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
