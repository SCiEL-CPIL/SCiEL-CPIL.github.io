import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    n, m, p = DIMENSIONS
    
    rank = 43
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k, c):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
        
    def add_strassen(gi0, gj0, gk0):
        nonlocal col
        # M1
        U[gi0 * m + gj0][col] = 1
        U[(gi0+1) * m + (gj0+1)][col] = 1
        V[gj0 * p + gk0][col] = 1
        V[(gj0+1) * p + (gk0+1)][col] = 1
        W[gk0 * n + gi0][col] = 1
        W[(gk0+1) * n + gi0][col] = 1
        W[gk0 * n + (gi0+1)][col] = -1
        W[(gk0+1) * n + (gi0+1)][col] = 1
        col += 1
        
        # M2
        U[(gi0+1) * m + gj0][col] = 1
        U[(gi0+1) * m + (gj0+1)][col] = 1
        V[gj0 * p + gk0][col] = 1
        W[gk0 * n + (gi0+1)][col] = 1
        col += 1
        
        # M3
        U[gi0 * m + gj0][col] = 1
        V[(gj0+1) * p + (gk0+1)][col] = 1
        W[(gk0+1) * n + gi0][col] = -1
        W[(gk0+1) * n + (gi0+1)][col] = -1
        col += 1
        
        # M4
        U[(gi0+1) * m + (gj0+1)][col] = 1
        V[(gj0+1) * p + gk0][col] = 1
        V[gj0 * p + gk0][col] = 1
        W[gk0 * n + (gi0+1)][col] = 1
        W[(gk0+1) * n + (gi0+1)][col] = -1
        col += 1
        
        # M5
        U[gi0 * m + (gj0+1)][col] = 1
        V[(gj0+1) * p + (gk0+1)][col] = 1
        V[gj0 * p + (gk0+1)][col] = 1
        W[(gk0+1) * n + gi0][col] = -1
        W[(gk0+1) * n + (gi0+1)][col] = 1
        col += 1
        
        # M6
        U[(gi0+1) * m + gj0][col] = 1
        U[gi0 * m + gj0][col] = -1
        V[gj0 * p + (gk0+1)][col] = 1
        V[(gj0+1) * p + (gk0+1)][col] = 1
        W[gk0 * n + (gi0+1)][col] = 1
        W[(gk0+1) * n + gi0][col] = -1
        col += 1
        
        # M7
        U[gi0 * m + (gj0+1)][col] = 1
        U[(gi0+1) * m + (gj0+1)][col] = -1
        V[(gj0+1) * p + gk0][col] = 1
        V[gj0 * p + (gk0+1)][col] = 1
        W[gk0 * n + (gi0+1)][col] = -1
        W[(gk0+1) * n + gi0][col] = -1
        col += 1

    # Block 1: Rows 0,1; Cols 0,1; Depth 0,1
    add_strassen(0, 0, 0)
    
    # Block 2: Rows 0,1; Cols 0,1; Depth 2
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook(i, 2, k, col)
            col += 1
            
    # Block 3: Rows 0,1; Cols 2,3; Depth 0,1
    add_strassen(0, 0, 2)
    
    # Block 4: Rows 0,1; Cols 2,3; Depth 2
    for i in [0, 1]:
        for k in [2, 3]:
            add_schoolbook(i, 2, k, col)
            col += 1
            
    # Block 5: Rows 0,1; Col 4; Depth 0,1,2
    for i in [0, 1]:
        for j in [0, 1, 2]:
            add_schoolbook(i, j, 4, col)
            col += 1
            
    # Block 6: Row 2; Cols 0,1,2,3,4; Depth 0,1,2
    for k in [0, 1, 2, 3, 4]:
        for j in [0, 1, 2]:
            add_schoolbook(2, j, k, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}
