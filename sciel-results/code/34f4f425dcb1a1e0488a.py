import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = DIMENSIONS
    # We construct a decomposition by partitioning the k-index (columns of B, rows of C)
    # into groups of size 3, 3, and 1.
    # Group 1: k in [0, 3) -> 3x3x3 multiplication, identity rank 27
    # Group 2: k in [3, 6) -> 3x3x3 multiplication, identity rank 27
    # Group 3: k in [6, 7) -> 3x3x1 multiplication, identity rank 3
    # Total rank = 27 + 27 + 3 = 57.
    
    rank = 57
    
    u_rows_count = n * m
    v_rows_count = m * p
    w_rows_count = p * n
    
    U = [[0] * rank for _ in range(u_rows_count)]
    V = [[0] * rank for _ in range(v_rows_count)]
    W = [[0] * rank for _ in range(w_rows_count)]
    
    r = 0
    # Group 1: k = 0, 1, 2
    for k in range(3):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                r += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    # Group 2: k = 3, 4, 5
    for k in range(3, 6):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                r += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    # Group 3: k = 6
    for k in range(6, 7):
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
                r += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
