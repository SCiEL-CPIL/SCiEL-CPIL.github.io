import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10s margin, task limit 120s
    
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    blocks = [
        (range(0, 2), range(0, 2), range(0, 2)),
        (range(2, 4), range(2, 4), range(0, 2)),
        (range(0, 2), range(0, 2), range(2, 4)),
        (range(2, 4), range(2, 4), range(2, 4)),
    ]
    
    r = 0
    for A_js, B_js, C_ks in blocks:
        j0, j1 = A_js[0], A_js[1]
        k0, k1 = C_ks[0], C_ks[1]
        
        for i in range(n):
            U[i * m + j0][r] = 1
            V[j0 * p + k0][r] = 1
            W[k0 * n + i][r] = 1
        r += 1
        
        for i in range(n):
            U[i * m + j1][r] = 1
            V[j1 * p + k0][r] = 1
            W[k0 * n + i][r] = 1
        r += 1
        
        for i in range(n):
            U[i * m + j0][r] = 1
            V[j0 * p + k1][r] = 1
            W[k1 * n + i][r] = 1
        r += 1
        
        for i in range(n):
            U[i * m + j1][r] = 1
            V[j1 * p + k1][r] = 1
            W[k1 * n + i][r] = 1
        r += 1
        
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
