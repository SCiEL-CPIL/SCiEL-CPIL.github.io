"""Matrix multiplication 2x2 by 2x13 rank-minimal tensor decomposition.

Uses Strassen's algorithm for the first two columns (rank 7) and
schoolbook for the remaining 11 columns (4 each).

Total rank = 7 + 4*11 = 51.
"""

import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    Uses Strassen for first 2 columns (rank 7) + schoolbook for remaining 11 (4 each).
    Total rank = 7 + 44 = 51.
    """
    start_time = time.time()
    deadline = 12.0  # seconds
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # A is 2x2, B is 2x13, C is 2x13
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # B indices: j*p+k, so B[0,k]=k, B[1,k]=13+k
    # C indices (W): k*n+i, so C[0,k]=2k, C[1,k]=2k+1
    
    rank = 7 + 4 * (p - 2)  # 7 + 44 = 51
    U = [[0] * rank for _ in range(n * m)]  # 4 x 51
    V = [[0] * rank for _ in range(m * p)]  # 26 x 51
    W = [[0] * rank for _ in range(p * n)]  # 26 x 51
    
    # Strassen for first 2 columns (k=0,1)
    # a=A[0,0](0), b=A[0,1](1), c=A[1,0](2), d=A[1,1](3)
    # e=B[0,0](0), f=B[0,1](1), g=B[1,0](13), h=B[1,1](14)
    # C[0,0](0), C[0,1](2), C[1,0](1), C[1,1](3)
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[14][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[14][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4 = d(g-f)
    U[3][3] = 1
    V[13][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = -1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[14][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[13][6] = 1; V[14][6] = 1
    W[0][6] = 1
    
    # Schoolbook for columns k=2..12
    col = 7
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
