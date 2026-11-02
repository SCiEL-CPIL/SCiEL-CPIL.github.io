import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,1,3} (j=0)
    # C_{ik} = A_{i0} B_{0k}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2a: M_{2,2,1} (j in {1,2}, k=1)
    # C_{i1} = A_{i1}B_{11} + A_{i2}B_{21}
    # Local A: 2x2, Local B: 2x1 (col 0 of local B, which is global k=1)
    # Local C: 2x1 (col 0 of local C, which is global k=1)
    # Q1 = A_{0,1} B_{1,1}
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    # Q2 = A_{0,2} B_{2,1}
    U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 0][col] = 1
    col += 1
    # Q3 = A_{1,1} B_{1,1}
    U[1 * m + 1][col] = 1
    V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    # Q4 = A_{1,2} B_{2,1}
    U[1 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1

    # Part 2b: M_{2,2,2} (j in {1,2}, k in {1,2})
    # Strassen decomposition
    # Local A: 2x2 (rows 0,1; cols 1,2 of global A)
    # Local B: 2x2 (rows 1,2; cols 1,2 of global B)
    # Local C: 2x2 (rows 0,1; cols 1,2 of global C)
    
    # P1 = (A_{0,1} + A_{1,2}) (B_{1,1} + B_{2,2})
    U[0 * m + 1][col] = 1
    U[1 * m + 2][col] = 1
    V[1 * p + 1][col] = 1
    V[2 * p + 2][col] = 1
    W[1 * n + 0][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    
    # P2 = (A_{1,1} + A_{1,2}) B_{1,1}
    U[1 * m + 1][col] = 1
    U[1 * m + 2][col] = 1
    V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    W[2 * n + 1][col] = -1
    col += 1
    
    # P3 = A_{0,1} (B_{1,2} - B_{2,2})
    U[0 * m + 1][col] = 1
    V[1 * p + 2][col] = 1
    V[2 * p + 2][col] = -1
    W[1 * n + 0][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    
    # P4 = A_{1,2} (B_{2,1} - B_{1,1})
    U[1 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    V[1 * p + 1][col] = -1
    W[1 * n + 0][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # P5 = (A_{0,1} + A_{0,2}) B_{2,2}
    U[0 * m + 1][col] = 1
    U[0 * m + 2][col] = 1
    V[2 * p + 2][col] = 1
    W[1 * n + 0][col] = -1
    W[2 * n + 0][col] = 1
    col += 1
    
    # P6 = (A_{1,1} - A_{0,1}) (B_{1,1} + B_{1,2})
    U[1 * m + 1][col] = 1
    U[0 * m + 1][col] = -1
    V[1 * p + 1][col] = 1
    V[1 * p + 2][col] = 1
    W[2 * n + 1][col] = 1
    col += 1
    
    # P7 = (A_{0,2} - A_{1,2}) (B_{2,1} + B_{2,2})
    U[0 * m + 2][col] = 1
    U[1 * m + 2][col] = -1
    V[2 * p + 1][col] = 1
    V[2 * p + 2][col] = 1
    W[1 * n + 0][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
