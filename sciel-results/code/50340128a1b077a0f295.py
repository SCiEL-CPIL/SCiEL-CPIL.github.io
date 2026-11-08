import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # U index: i * m + k
                u_idx = i * m + k
                # V index: k * p + j
                v_idx = k * p + j
                # W index: j * n + i (assuming standard C[i,j] storage)
                # Task says "W[k*n+i,r]"? 
                # Let's check the seed code's W indexing.
                # Seed: W is (p*n, rank).
                # In seed, c00 = k_start * 2 + 0.
                # k_start is row of B? No, k_start is col of B?
                # In seed: b00 = 0*7 + k_start. B is 3x7. Row 0, Col k_start.
                # c00 = k_start * 2 + 0. C is 2x7? No, C is 2x7?
                # n=2, p=7. C is 2x7.
                # c00 index 0. c01 index 1.
                # If C is 2x7, flattened row-major: C[i, j] is i*7 + j.
                # Seed uses c00 = k_start * 2 + 0.
                # This implies C is stored column-major? Or C is 7x2?
                # "W[k*n+i,r]"
                # If W index is k*n+i, and k is from B (0..m-1)? No, W is output.
                # Let's look at the seed's W usage.
                # add_term(..., [c00, c11], ...)
                # c00 = 0. c11 = 3.
                # If C is 2x7, indices 0..13.
                # If C is 7x2, indices 0..13.
                # Seed: n=2, p=7.
                # "W[k*n+i,r]"
                # If i is row of C (0..1), k is col of C (0..6)?
                # Then index = k*2 + i.
                # This matches c00 = 0*2+0=0, c01 = 0*2+1=1? No, c01 = 1.
                # c10 = 1*2+0=2. c11 = 1*2+1=3.
                # So W index = col_C * n + row_C.
                # This is column-major storage for C (2x7).
                
                w_idx = j * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
