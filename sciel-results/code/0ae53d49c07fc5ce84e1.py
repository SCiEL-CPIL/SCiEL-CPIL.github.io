# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition (Schoolbook)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
                
    # Part 2: Schoolbook for i in {0,1}, j in {0,1}, k=2
    for i in range(2):
        for j in range(2):
            k = 2
            U[i*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    # Part 3: Schoolbook for i in {0,1}, j=2, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            j = 2
            U[i*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
