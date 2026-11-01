import time

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Rank 46 decomposition
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Strassen blocks
    for block in range(6):
        if time.time() > deadline:
            break
        k_start = block * 2
        k1 = k_start
        k2 = k_start + 1
        
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k2 * n + 0
        w_10 = k1 * n + 1
        w_11 = k2 * n + 1
        
        # M1
        U[0][col] += 1; U[3][col] += 1
        V[v_00][col] += 1; V[v_11][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1
        
        # M2
        U[2][col] += 1; U[3][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1
        
        # M3
        U[0][col] += 1
        V[v_01][col] += 1; V[v_11][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1
        
        # M4
        U[3][col] += 1
        V[v_10][col] += 1; V[v_00][col] += -1
        W[w_00][col] += -1; W[w_10][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        col += 1
        
        # M6
        U[2][col] += 1; U[0][col] += -1
        V[v_00][col] += 1; V[v_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[3][col] += -1
        V[v_10][col] += 1; V[v_11][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

    # Last block
    if time.time() <= deadline:
        k = 12
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        w_0 = k * n + 0
        w_1 = k * n + 1
        
        # T1
        U[0][col] += 1
        V[v_0][col] += 1
        W[w_0][col] += 1
        col += 1
        
        # T2
        U[1][col] += 1
        V[v_1][col] += 1
        W[w_0][col] += 1
        col += 1
        
        # T3
        U[2][col] += 1
        V[v_0][col] += 1
        W[w_1][col] += 1
        col += 1
        
        # T4
        U[3][col] += 1
        V[v_1][col] += 1
        W[w_1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
