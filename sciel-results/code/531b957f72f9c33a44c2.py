"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Split middle dimension (3) into 2 + 1
    # Split last dimension (10) into 5 groups of 2
    # For each 2x2 x 2x2 block: use Strassen (rank 7)
    # For the 2x1 x 1x10 part: rank 2
    # Total rank = 5*7 + 2 = 37
    
    rank = 5 * 7 + 2  # 37
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factor matrices for 2x2 x 2x2
    strassen_U = [
        [1, 0, 0, 1],  # m1 = (a+d)(e+h)
        [0, 0, 1, 1],  # m2 = (c+d)e
        [1, 0, 0, 0],  # m3 = a(f-h)
        [0, 0, 0, 1],  # m4 = d(g-e)
        [1, 1, 0, 0],  # m5 = (a+b)h
        [-1, 0, 1, 0], # m6 = (c-a)(e+f)
        [0, 1, 0, -1], # m7 = (b-d)(g+h)
    ]
    strassen_V = [
        [1, 1, 0, 0],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1], # m3
        [-1, 0, 1, 0], # m4
        [0, 0, 0, 1],  # m5
        [1, 1, 0, 0],  # m6
        [0, 0, 1, 1],  # m7
    ]
    strassen_W = [
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 0],  # m2
        [0, 1, 0, 0],  # m3
        [1, 0, 1, 0],  # m4
        [-1, 1, 0, 0], # m5
        [0, 0, 0, 1],  # m6
        [1, 0, 0, 0],  # m7
    ]
    
    col = 0
    for block in range(5):
        if time.time() > deadline:
            break
        for s in range(7):
            for ai in range(2):
                for aj in range(2):
                    u_idx = ai * m + aj
                    U[u_idx][col] += strassen_U[s][ai * 2 + aj]
            
            for bj in range(2):
                for bk in range(2):
                    v_idx = bj * p + (2 * block + bk)
                    V[v_idx][col] += strassen_V[s][bj * 2 + bk]
            
            for wi in range(2):
                for wk in range(2):
                    w_idx = (2 * block + wk) * n + wi
                    W[w_idx][col] += strassen_W[s][wi * 2 + wk]
            
            col += 1
    
    # Handle 2x1 x 1x10 part
    for i in range(2):
        if time.time() > deadline:
            break
        u_idx = i * m + 2
        U[u_idx][col] = 1
        for k in range(10):
            v_idx = 2 * p + k
            V[v_idx][col] = 1
            w_idx = k * n + i
            W[w_idx][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
