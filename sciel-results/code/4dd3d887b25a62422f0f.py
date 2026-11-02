import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition vectors
    # u: [A11, A12, A21, A22]
    # v: [B11, B12, B21, B22]
    # w: [C11, C12, C21, C22]
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    # M2: (A21+A22)B11 -> C21
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    
    # M3: A11(B12-B22) -> C12, C22
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # M4: A22(B21-B11) -> C11, C21
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    
    # M5: (A11+A12)B22 -> C11(-1), C12
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # M6: (A21-A11)(B11+B12) -> C22
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # M7: (A12-A22)(B21+B22) -> C11
    u7 = [0, 1, -1, 0]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    strassen_vectors = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    col = 0
    # 5 blocks of 2 columns in B
    for t in range(5):
        for s in range(7):
            u_s, v_s, w_s = strassen_vectors[s]
            
            # Map U
            # Local 0 (A11) -> Global 0
            if u_s[0] != 0: U[0][col] = u_s[0]
            # Local 1 (A12) -> Global 1
            if u_s[1] != 0: U[1][col] = u_s[1]
            # Local 2 (A21) -> Global 3
            if u_s[2] != 0: U[3][col] = u_s[2]
            # Local 3 (A22) -> Global 4
            if u_s[3] != 0: U[4][col] = u_s[3]
            
            # Map V
            # Local 0 (B11) -> Global 2t
            if v_s[0] != 0: V[2*t][col] = v_s[0]
            # Local 1 (B12) -> Global 2t+1
            if v_s[1] != 0: V[2*t+1][col] = v_s[1]
            # Local 2 (B21) -> Global 10+2t
            if v_s[2] != 0: V[10+2*t][col] = v_s[2]
            # Local 3 (B22) -> Global 10+2t+1
            if v_s[3] != 0: V[10+2*t+1][col] = v_s[3]
            
            # Map W
            # Local 0 (C11) -> Global 4t
            if w_s[0] != 0: W[4*t][col] = w_s[0]
            # Local 1 (C12) -> Global 4t+2
            if w_s[1] != 0: W[4*t+2][col] = w_s[1]
            # Local 2 (C21) -> Global 4t+1
            if w_s[2] != 0: W[4*t+1][col] = w_s[2]
            # Local 3 (C22) -> Global 4t+3
            if w_s[3] != 0: W[4*t+3][col] = w_s[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Remaining 2x1x10 part
    # A col 2 (indices 2, 5 in U)
    # B row 2 (indices 20-29 in V)
    # C all (indices 0-19 in W)
    
    for i in range(2):
        for k in range(10):
            # U index
            u_idx = i * 3 + 2
            U[u_idx][col] = 1
            
            # V index
            v_idx = 20 + k
            V[v_idx][col] = 1
            
            # W index
            w_idx = k * 2 + i
            W[w_idx][col] = 1
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
