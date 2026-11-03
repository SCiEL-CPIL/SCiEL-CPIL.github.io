import time

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0

def _time_ok():
    return time.time() < _DEADLINE

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            u_idx = j
            v_idx = j * p + k
            w_idx = k * n
            
            for i in range(n):
                U[u_idx][col] = 1
                W[w_idx + i][col] = 1
            
            V[v_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
