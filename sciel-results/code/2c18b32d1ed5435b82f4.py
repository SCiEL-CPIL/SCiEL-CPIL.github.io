import time

DIMENSIONS = [2, 3, 4]

def construct():
    # Start time to respect the 120s (or 150s) limit, though this runs in <1ms
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24 for 2x3, 3x4
    
    # Shapes:
    # U: (n*m, rank) -> (6, 24)
    # V: (m*p, rank) -> (12, 24)
    # W: (p*n, rank) -> (8, 24)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize matrices with zeros
    # Using lists of lists as per the interface requirement for integer/half-integer entries
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                idx_u = i * m + j
                # Index in V: row j, col k -> j*p + k
                idx_v = j * p + k
                # Index in W: output (i, k). Convention W[k*n + i, r]
                idx_w = k * n + i
                
                U[idx_u][r] = 1
                V[idx_v][r] = 1
                W[idx_w][r] = 1
                
                r += 1
                
    # Verify the solution quickly (optional, but good for safety)
    # The schoolbook decomposition is exact by construction.
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) matches, else 0.
    # Since each (i,j,k) corresponds to a unique column r, the product is 1 for that specific triple and 0 for others.
    
    return {"U": U, "V": V, "W": W}
