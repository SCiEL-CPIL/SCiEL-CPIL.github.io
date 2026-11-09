import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    rank = n * m * p  # 60
    
    # U shape: (n*m, rank) -> (6, 60)
    # V shape: (m*p, rank) -> (30, 60)
    # W shape: (p*n, rank) -> (20, 60)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Fallback: return what we have. 
                    # Since we are doing it in a tight loop, this shouldn't trigger for 60 iterations.
                    return {"U": U, "V": V, "W": W}
                
                # Index mapping:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i  (Official trace convention W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
