"""Rewritten program: rank-22 decomposition via 2x2 Strassen blocks + schoolbook."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
DEADLINE = 120.0

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0.0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 22
    
    col = 0
    
    # 2x2 Strassen for A[:,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C indices: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2x2 Strassen for A[:,0:2] x B[0:2,2:4] -> C[0:2,2:4]
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C indices: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # M6
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # A[:,2] x B[2,:] -> C[0:2,0:4]
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    
    for k in range(4):
        U[2][col] = 1
        V[8+k][col] = 1
        W[k*2][col] = 1
        col += 1
    
    for k in range(4):
        U[5][col] = 1
        V[8+k][col] = 1
        W[k*2+1][col] = 1
        col += 1
    
    # Bottom row: A[2,:] x B[0:3,:] -> C[2,0:4]
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    for k in range(4):
        for j in range(3):
            U[6+j][col] = 1
            V[j*4+k][col] = 1
            W[k*2+2][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    # Numerical verification
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                val = 0
                for cc in range(rank):
                    val += U_np[a,cc] * V_np[b,cc] * W_np[c,cc]
                assert abs(val - 1) < 1e-9, f"Failed for i={i},j={j},k={k}: got {val}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
