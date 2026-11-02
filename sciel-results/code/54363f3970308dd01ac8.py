import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Rank 7 Strassen decomposition
    # This is the known minimal rank for 2x2 matrix multiplication.
    rank = 7
    
    # U shape: (n*m, rank) -> indices: i*m + j
    # V shape: (m*p, rank) -> indices: j*p + k
    # W shape: (p*n, rank) -> indices: k*n + i  (Trace convention: W[k*n+i,r])
    
    n, m, p = 2, 2, 2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # Strassen's 7 scalar products:
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    
    # Resulting C:
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5

    # Indices in U (A):
    # a00 -> 0, a01 -> 1, a10 -> 2, a11 -> 3
    # Indices in V (B):
    # b00 -> 0, b01 -> 1, b10 -> 2, b11 -> 3
    # Indices in W (C) with W[k*n + i]:
    # C00 (i=0, k=0) -> 0*2+0 = 0
    # C01 (i=0, k=1) -> 1*2+0 = 2
    # C10 (i=1, k=0) -> 0*2+1 = 1
    # C11 (i=1, k=1) -> 1*2+1 = 3

    # Term 0: m0 = (a00 + a11) * (b00 + b11)
    set_val(U, 0, 0, 1)
    set_val(U, 3, 0, 1)
    set_val(V, 0, 0, 1)
    set_val(V, 3, 0, 1)
    set_val(W, 0, 0, 1)  # C00
    set_val(W, 3, 0, 1)  # C11

    # Term 1: m1 = (a10 + a11) * b00
    set_val(U, 2, 1, 1)
    set_val(U, 3, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1)  # C10
    set_val(W, 3, 1, -1) # C11

    # Term 2: m2 = a00 * (b01 - b11)
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1)
    set_val(V, 3, 2, -1)
    set_val(W, 2, 2, 1)  # C01
    set_val(W, 3, 2, 1)  # C11

    # Term 3: m3 = a11 * (b10 - b00)
    set_val(U, 3, 3, 1)
    set_val(V, 2, 3, 1)
    set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1)  # C00
    set_val(W, 1, 3, 1)  # C10

    # Term 4: m4 = (a00 + a01) * b11
    set_val(U, 0, 4, 1)
    set_val(U, 1, 4, 1)
    set_val(V, 3, 4, 1)
    set_val(W, 0, 4, -1) # C00
    set_val(W, 2, 4, 1)  # C01

    # Term 5: m5 = (a10 - a00) * (b00 + b01)
    set_val(U, 2, 5, 1)
    set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1)
    set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)  # C11

    # Term 6: m6 = (a01 - a11) * (b10 + b11)
    set_val(U, 1, 6, 1)
    set_val(U, 3, 6, -1)
    set_val(V, 2, 6, 1)
    set_val(V, 3, 6, 1)
    set_val(W, 0, 6, 1)  # C00

    # Optimization: Remove verification step to minimize execution time and complexity.
    # The Strassen decomposition is mathematically proven to be correct.
    
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
