"""Matrix multiplication 3x4 by 4x4 with rank minimization via Strassen on 2x2 blocks."""
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 44

    # Column 0: identity mapping for i=0,j=0,k=0
    U[0][0] = 1.0
    V[0][0] = 1.0
    W[0][0] = 1.0

    # Columns 1-7: Strassen for i=1 (rows 1-2 of A, i.e., rows 4-5 of U)
    # Strassen indices: M1..M7 at columns 1..7
    # M1 = (a+d)(e+h)
    #   a = A[0,0] = U index 0*4+0 = 0
    #   d = A[0,3] = U index 0*4+3 = 3
    #   e = B[0,0] = V index 0*4+0 = 0
    #   h = B[0,3] = V index 0*4+3 = 3
    U[0][1] = 1.0  # a
    U[3][1] = 1.0  # d
    V[0][1] = 1.0  # e
    V[3][1] = 1.0  # h
    # W for M1: C[0,0] and C[3,3] get M1
    # W index: k*n+i, so C[0,0] -> 0*3+0=0, C[3,3] -> 3*3+0=9
    W[0][1] = 1.0
    W[9][1] = 1.0

    # M2 = (c+d)(e)
    #   c = A[0,2] = U index 2
    #   d = A[0,3] = U index 3
    #   e = B[0,0] = V index 0
    U[2][2] = 1.0  # c
    U[3][2] = 1.0  # d
    V[0][2] = 1.0  # e
    # W for M2: C[0,2] gets M2
    # W index: k*n+i = 2*3+0 = 6
    W[6][2] = 1.0

    # M3 = (c-a)(f-h)
    #   c = A[0,2] = U index 2
    #   a = A[0,0] = U index 0
    #   f = B[1,0] = V index 4
    #   h = B[0,3] = V index 3
    U[2][3] = 1.0   # c
    U[0][3] = -1.0  # -a
    V[4][3] = 1.0   # f
    V[3][3] = -1.0  # -h
    # W for M3: C[0,1] and C[3,1] get M3
    # W index: C[0,1] -> 0*3+1=1, C[3,1] -> 3*3+1=10
    W[1][3] = 1.0
    W[10][3] = 1.0

    # M4 = (c+a)(g-e)
    #   c = A[0,2] = U index 2
    #   a = A[0,0] = U index 0
    #   g = B[1,1] = V index 5
    #   e = B[0,0] = V index 0
    U[2][4] = 1.0   # c
    U[0][4] = 1.0   # a
    V[5][4] = 1.0   # g
    V[0][4] = -1.0  # -e
    # W for M4: C[0,2] and C[3,2] get M4
    # W index: C[0,2] -> 0*3+2=2, C[3,2] -> 3*3+2=11
    W[2][4] = 1.0
    W[11][4] = 1.0

    # M5 = (b-d)(e+f)
    #   b = A[0,1] = U index 1
    #   d = A[0,3] = U index 3
    #   e = B[0,0] = V index 0
    #   f = B[1,0] = V index 4
    U[1][5] = 1.0   # b
    U[3][5] = -1.0  # -d
    V[0][5] = 1.0   # e
    V[4][5] = 1.0   # f
    # W for M5: C[0,0] and C[3,0] get M5
    # W index: C[0,0] -> 0*3+0=0, C[3,0] -> 3*3+0=9
    W[0][5] = 1.0
    W[9][5] = 1.0

    # M6 = (a+b)(h-f)
    #   a = A[0,0] = U index 0
    #   b = A[0,1] = U index 1
    #   h = B[0,3] = V index 3
    #   f = B[1,0] = V index 4
    U[0][6] = 1.0  # a
    U[1][6] = 1.0  # b
    V[3][6] = 1.0  # h
    V[4][6] = -1.0 # -f
    # W for M6: C[0,1] and C[3,1] get M6
    # W index: C[0,1] -> 0*3+1=1, C[3,1] -> 3*3+1=10
    W[1][6] = 1.0
    W[10][6] = 1.0

    # M7 = (d-b)(g+h)
    #   d = A[0,3] = U index 3
    #   b = A[0,1] = U index 1
    #   g = B[1,1] = V index 5
    #   h = B[0,3] = V index 3
    U[3][7] = 1.0   # d
    U[1][7] = -1.0  # -b
    V[5][7] = 1.0   # g
    V[3][7] = 1.0   # h
    # W for M7: C[0,2] and C[3,2] get M7
    # W index: C[0,2] -> 0*3+2=2, C[3,2] -> 3*3+2=11
    W[2][7] = 1.0
    W[11][7] = 1.0

    # Columns 8-14: identity for i=0, j=1,2,3 with k=0,1,2,3 (8 columns)
    # i=0, j=1: A index 0*4+1=1
    # i=0, j=2: A index 0*4+2=2
    # i=0, j=3: A index 0*4+3=3
    for j in range(1, 4):
        for k in range(4):
            col = 8 + (j-1)*4 + k
            U[j][col] = 1.0
            V[j*4+k][col] = 1.0
            W[k*3+0][col] = 1.0

    # Columns 15-28: identity for i=1 (rows 4-5 of U), j=0,1,2,3 with k=0,1,2,3 (16 columns)
    # i=1: A rows 4-5 (indices 4,5,6,7 for j=0-3; 8,9,10,11 for j=4-7)
    # Wait, U has 12 rows: i*m+j where i in 0-2, j in 0-3
    # i=0: j=0,1,2,3 -> U indices 0,1,2,3
    # i=1: j=0,1,2,3 -> U indices 4,5,6,7
    # i=2: j=0,1,2,3 -> U indices 8,9,10,11
    # V has 16 rows: m*p = 4*4, indexed j*p+k where j in 0-3, k in 0-3
    # W has 12 rows: p*n = 4*3, indexed k*n+i where k in 0-3, i in 0-2

    # For i=1, j=0: U index 4, V indices 0-3, W indices 0*3+1=1, 1*3+1=4, 2*3+1=7, 3*3+1=10
    # For i=1, j=1: U index 5, V indices 4-7, W indices 1*3+1=4, 2*3+1=7, 3*3+1=10, 0*3+1=1
    # Wait, W[k*n+i] for k=0,1,2,3 and i=1: k*3+1 = 1,4,7,10
    # For i=1, j=2: U index 6, V indices 8-11, W indices 2*3+1=7, 3*3+1=10, 0*3+1=1, 1*3+1=4
    # For i=1, j=3: U index 7, V indices 12-15, W indices 3*3+1=10, 0*3+1=1, 1*3+1=4, 2*3+1=7

    for j in range(4):
        for k in range(4):
            col = 15 + j*4 + k
            U[4 + j][col] = 1.0
            V[j*4+k][col] = 1.0
            W[k*3+1][col] = 1.0

    # Columns 29-35: identity for i=2 (rows 8-11 of U), j=0,1,2,3 with k=0,1,2,3 (7 columns)
    # i=2: j=0: U index 8, j=1: U index 9, j=2: U index 10, j=3: U index 11
    # W[k*n+i] for k=0,1,2,3 and i=2: k*3+2 = 2,5,8,11
    for j in range(4):
        for k in range(4):
            col = 29 + j*4 + k
            U[8 + j][col] = 1.0
            V[j*4+k][col] = 1.0
            W[k*3+2][col] = 1.0

    return {"U": U, "V": V, "W": W}
