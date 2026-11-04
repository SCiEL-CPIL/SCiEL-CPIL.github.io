import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for this
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # 0: A11 B11
    # 1: A12 B21
    # 2: A11 B12
    # 3: A12 B22
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            if time.time() > deadline:
                break
            add_strassen_step(U, V, W, col, b, s)
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}

def add_strassen_step(U, V, W, col, block_type, step):
    # ... (implementation as above)
    pass
