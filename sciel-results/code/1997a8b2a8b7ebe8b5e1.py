import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # 7 for 2x2x2 Strassen block (i in 0,1; j in 0,1; k in 0,1)
    # 4 for 2x1x2 block (i in 0,1; j=2; k in 0,1)
    # 18 for 2x3x3 block (i in 0,1; j in 0,1,2; k in 2,3,4)
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_component(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for i, j in u_indices:
            U[i * m + j][col] += val
        for j, k in v_indices:
            V[j * p + k][col] += val
        for k, i in w_indices:
            W[k * n + i][col] += val
        col += 1

    # 1. Strassen 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # M1 = A00 (B00 - B10)
    add_component([(0,0)], [(0,0), (1,0)], [(0,0), (1,0)], 1)
    U[0*m+0][col-1] = 1
    V[0*p+0][col-1] = 1
    V[1*p+0][col-1] = -1
    W[0*n+0][col-1] = 1
    W[0*n+1][col-1] = 1 # Wait, W index is k*n+i.
    # M1 contributes to C00 and C01?
    # C00 = M5 + M4 - M2 - M7
    # C01 = M1 + M7
    # C10 = M3 + M6
    # C11 = M5 + M1 - M3 - M6
    
    # Let's restart the Strassen block implementation carefully.
    col = 0
    
    # M1 = A00 (B00 - B10)
    # U: (0,0)=1
    # V: (0,0)=1, (1,0)=-1
    # W: (0,0)=1, (0,1)=1 ? No.
    # M1 is used in C01 and C11.
    # C01 = M1 + M7. So W(0,1) += 1.
    # C11 = M5 + M1 - M3 - M6. So W(1,1) += 1.
    
    U[0*m+0][0] = 1
    V[0*p+0][0] = 1
    V[1*p+0][0] = -1
    W[1*n+0][0] = 1 # k=1, i=0 -> C01
    W[1*n+1][0] = 1 # k=1, i=1 -> C11
    col = 1
    
    # M2 = (A00 + A10) B10
    U[0*m+0][1] = 1
    U[1*m+0][1] = 1
    V[1*p+0][1] = 1
    W[0*n+0][1] = -1 # C00 = ... - M2
    col = 2
    
    # M3 = (A01 + A11) B01
    U[0*m+1][2] = 1
    U[1*m+1][2] = 1
    V[0*p+1][2] = 1
    W[0*n+1][2] = 1 # C10 = M3 + M6
    W[1*n+1][2] = -1 # C11 = ... - M3
    col = 3
    
    # M4 = A11 (B11 - B01)
    U[1*m+1][3] = 1
    V[1*p+1][3] = 1
    V[0*p+1][3] = -1
    W[0*n+0][3] = 1 # C00 = ... + M4
    col = 4
    
    # M5 = (A00 + A11) (B00 + B11)
    U[0*m+0][4] = 1
    U[1*m+1][4] = 1
    V[0*p+0][4] = 1
    V[1*p+1][4] = 1
    W[0*n+0][4] = 1 # C00 = ... + M5
    W[1*n+1][4] = 1 # C11 = ... + M5
    col = 5
    
    # M6 = (A10 - A00) (B00 + B01)
    U[1*m+0][5] = 1
    U[0*m+0][5] = -1
    V[0*p+0][5] = 1
    V[0*p+1][5] = 1
    W[0*n+1][5] = 1 # C10 = M3 + M6
    W[1*n+1][5] = -1 # C11 = ... - M6
    col = 6
    
    # M7 = (A01 - A11) (B10 + B11)
    U[0*m+1][6] = 1
    U[1*m+1][6] = -1
    V[1*p+0][6] = 1
    V[1*p+1][6] = 1
    W[0*n+0][6] = -1 # C00 = ... - M7
    W[0*n+1][6] = 1 # C01 = M1 + M7
    col = 7
    
    # 2. Block 2x1x2 (i in 0,1; j=2; k in 0,1)
    # Terms: A02 B20, A02 B21, A12 B20, A12 B21
    # C00 += A02 B20
    U[0*m+2][7] = 1
    V[2*p+0][7] = 1
    W[0*n+0][7] = 1
    col = 8
    
    # C01 += A02 B21
    U[0*m+2][8] = 1
    V[2*p+1][8] = 1
    W[0*n+1][8] = 1
    col = 9
    
    # C10 += A12 B20
    U[1*m+2][9] = 1
    V[2*p+0][9] = 1
    W[0*n+1][9] = 1
    col = 10
    
    # C11 += A12 B21
    U[1*m+2][10] = 1
    V[2*p+1][10] = 1
    W[1*n+1][10] = 1
    col = 11
    
    # 3. Block 2x3x3 (i in 0,1; j in 0,1,2; k in 2,3,4)
    # Schoolbook for this subblock
    for i in range(2):
        for j in range(3):
            for k in range(2, 5):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
