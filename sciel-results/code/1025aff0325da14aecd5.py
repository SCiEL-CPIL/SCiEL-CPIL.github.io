import time

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for static construction
    
    n, m, p = DIMENSIONS
    rank = 46
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    strassen_blocks = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    for b in range(6):
        if time.time() > deadline:
            break
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_blocks[s]
            
            U[0][col] = u_loc[0]
            U[1][col] = u_loc[1]
            U[2][col] = u_loc[2]
            U[3][col] = u_loc[3]
            
            V[0 * p + k0][col] = v_loc[0]
            V[0 * p + k1][col] = v_loc[1]
            V[1 * p + k0][col] = v_loc[2]
            V[1 * p + k1][col] = v_loc[3]
            
            W[k0 * n + 0][col] = w_loc[0]
            W[k1 * n + 0][col] = w_loc[1]
            W[k0 * n + 1][col] = w_loc[2]
            W[k1 * n + 1][col] = w_loc[3]
            
            col += 1
            
    if time.time() < deadline:
        k0 = 12
        
        U[0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        U[1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[3][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
