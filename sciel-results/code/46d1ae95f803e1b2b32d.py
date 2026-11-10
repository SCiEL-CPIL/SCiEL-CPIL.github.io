import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 by 5x5 matrix.
    C = A * B where A is 4x5, B is 5x5, C is 4x5.
    Tensor T[i,j,k] = C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: Partition j-index into j1={0,1} and j2={2,3,4}
    Partition k-index into k1={0,1} and k2={2,3,4}
    
    Four sub-problems:
    1) i in 0-3, j in 0-1, k in 0-1: 4x2x2 = 8 terms (direct)
    2) i in 0-3, j in 0-1, k in 2-4: 4x2x3 = 12 terms (direct)
    3) i in 0-3, j in 2-4, k in 0-1: 4x3x2 = 12 terms (direct)
    4) i in 0-3, j in 2-4, k in 2-4: 4x3x3 = 36 terms (direct)
    Total: 68 terms
    
    Can we do better with Strassen on sub-blocks?
    For 4) we have 4 rows of A, 3 cols of B, 3 cols of C.
    We can treat each row of A separately and use Strassen for 3x3:
    Row i: A[i,2:5] (1x3) * B[2:5,2:5] (3x3) = C[i,2:5] (1x3)
    This is rank 3 per row, so 12 total for all 4 rows.
    But we can share the B factors...
    
    Actually, let's just use the direct approach for now and see if we can optimize.
    """
    n, m, p = 4, 5, 5
    start_time = time.time()
    deadline = start_time + 110.0
    
    U_rows = [[0] * 100 for _ in range(n * m)]  # Pre-allocate max
    V_rows = [[0] * 100 for _ in range(m * p)]
    W_rows = [[0] * 100 for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries.items():
            U_rows[idx][col] += val
        for idx, val in v_entries.items():
            V_rows[idx][col] += val
        for idx, val in w_entries.items():
            W_rows[idx][col] += val
        col += 1
    
    j1 = [0, 1]
    j2 = [2, 3, 4]
    k1 = [0, 1]
    k2 = [2, 3, 4]
    
    # 1) i in 0-3, j in 0-1, k in 0-1: 8 terms
    for i in range(n):
        for j in j1:
            for k in k1:
                add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # 2) i in 0-3, j in 0-1, k in 2-4: 12 terms
    for i in range(n):
        for j in j1:
            for k in k2:
                add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # 3) i in 0-3, j in 2-4, k in 0-1: 12 terms
    for i in range(n):
        for j in j2:
            for k in k1:
                add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # 4) i in 0-3, j in 2-4, k in 2-4: 36 terms (direct)
    # Can we use Strassen? A[i,2:5] is 1x3, B[2:5,2:5] is 3x3, C[i,2:5] is 1x3
    # For a single row: sum over 3 j's, 3 k's = 9 terms
    # But we can factor: for fixed i, C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a rank-3 bilinear form for each i
    # Total for 4 rows: 4*3*3 = 36, but we can share B terms
    # Actually no, each (i,j,k) is independent in the tensor
    for i in range(n):
        for j in j2:
            for k in k2:
                add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # Trim
    U = [row[:col] for row in U_rows]
    V = [row[:col] for row in V_rows]
    W = [row[:col] for row in W_rows]
    
    return {"U": U, "V": V, "W": W}
