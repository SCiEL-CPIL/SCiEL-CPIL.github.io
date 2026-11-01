import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Safety check: if we somehow exceeded time (unlikely for trivial), return what we have
    # But trivial is O(1) essentially.
    
    return {"U": U, "V": V, "W": W}
