import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    R = 12
    
    # Initialize matrices
    # U: (n*m) x R
    # V: (m*p) x R
    # W: (p*n) x R
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Trivial rank-12 decomposition
    # For each (i,j,k), add a rank-1 component
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][idx] = 1
                V[v_idx][idx] = 1
                W[w_idx][idx] = 1
                idx += 1
            
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
