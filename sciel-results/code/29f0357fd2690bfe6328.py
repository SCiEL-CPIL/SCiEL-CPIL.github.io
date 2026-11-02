import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 2, 2, 12
    nm, mp, pn = n * m, m * p, p * n  # 4, 24, 24
    
    # Build the matrix multiplication tensor as a matrix
    # T[a, (b, c)] = 1 if C[c] = A[a] * B[b] contributes
    T = np.zeros((nm, mp * pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b * pn + c] = 1
    
    # Try to find a low-rank exact decomposition
    # Use iterative rank-1 extraction with rounding
    
    best_rank = nm * mp  # upper bound (schoolbook)
    best_U, best_V, best_W = None, None, None
    
    # Directly construct the Strassen-based decomposition
    # Rank = (p // 2) * 7
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U: A indices (0:00, 1:01, 2:10, 3:11)
    # V: B indices (0:00, 1:01, 2:10, 3:11)
    # W: C indices (0:00, 1:01, 2:10, 3:11)
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1}),
        ({0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1}),
        ({0: -1, 2: 1},{0: 1, 1: 1},  {3: 1}),
        ({1: 1, 3: -1},{2: 1, 3: 1},  {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_c, v_c, w_c = strassen[s]
            
            for a_idx, val in u_c.items():
                U[a_idx][col] = val
            
            for local_b, val in v_c.items():
                if local_b == 0:
                    gb = 2 * t
                elif local_b == 1:
                    gb = 2 * t + 1
                elif local_b == 2:
                    gb = m * p // 2 + 2 * t
                else:
                    gb = m * p // 2 + 2 * t + 1
                V[gb][col] = val
            
            for local_c, val in w_c.items():
                if local_c == 0:
                    gw = (2 * t) * n + 0
                elif local_c == 1:
                    gw = (2 * t + 1) * n + 0
                elif local_c == 2:
                    gw = (2 * t) * n + 1
                else:
                    gw = (2 * t + 1) * n + 1
                W[gw][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
