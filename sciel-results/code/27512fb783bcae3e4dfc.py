import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 83
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for i in {0,1}, j in {0,1}, k in {0,1}
    # Term 0: M1
    U[0][col] = 1
    U[5][col] = 1
    V[0][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # Term 1: M2
    U[4][col] = 1
    U[5][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # Term 2: M3
    U[0][col] = 1
    V[1][col] = 1
    V[8][col] = -1
    W[3][col] = 1
    W[4][col] = -1
    col += 1
    
    # Term 3: M4
    U[5][col] = 1
    V[7][col] = 1
    V[0][col] = -1
    W[1][col] = 1
    W[0][col] = -1
    col += 1
    
    # Term 4: M5
    U[0][col] = 1
    U[1][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Term 5: M6
    U[4][col] = 1
    U[0][col] = -1
    V[0][col] = 1
    V[1][col] = 1
    W[4][col] = -1
    col += 1
    
    # Term 6: M7
    U[1][col] = 1
    U[5][col] = -1
    V[7][col] = 1
    V[8][col] = 1
    W[3][col] = 1
    W[0][col] = -1
    col += 1
    
    # Remaining schoolbook terms
    for i in [0, 1]:
        for j in [0, 1]:
            for k in range(2, 7):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    for i in [0, 1]:
        for j in [2, 3]:
            for k in range(7):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    for i in [2]:
        for j in range(4):
            for k in range(7):
                if time.time() > deadline:
                    break
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
