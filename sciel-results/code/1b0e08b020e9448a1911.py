"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Rank 29: Strassen on 2x2x2 sub-block + schoolbook for rest
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 15 x 29
    W = [[0] * rank for _ in range(p * n)]  # 10 x 29
    
    col = 0
    # Strassen 2x2x2 for A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # A indices: U[i*m+j] = U[i*3+j], i in {0,1}, j in {0,1}
    # B indices: V[j*p+k] = V[j*5+k], j in {0,1}, k in {0,1}
    # C indices: W[k*n+i] = W[k*2+i], k in {0,1}, i in {0,1}
    
    # m1 = (a1+a4)(b1+b4)
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1; W[0*2+1][col] = -1; W[1*2+0][col] = 1
    col += 1  # col=1
    
    # m2 = (a3+a4)b1
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*5+0][col] = 1
    W[1*2+0][col] = 1; W[1*2+1][col] = -1
    col += 1  # col=2
    
    # m3 = a1(b2-b4)
    U[0*3+0][col] = 1
    V[1*5+0][col] = 1; V[0*5+1][col] = -1
    W[0*2+1][col] = 1
    col += 1  # col=3
    
    # m4 = a4(b3-b1)
    U[1*3+1][col] = 1
    V[1*5+0][col] = 1; V[0*5+0][col] = -1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1
    col += 1  # col=4
    
    # m5 = (a1+a2)b4
    U[0*3+0][col] = 1; U[0*3+1][col] = 1
    V[1*5+1][col] = 1
    W[0*2+0][col] = -1; W[0*2+1][col] = 1
    col += 1  # col=5
    
    # m6 = (a3-a1)(b1+b2)
    U[1*3+0][col] = 1; U[0*3+0][col] = -1
    V[0*5+0][col] = 1; V[1*5+0][col] = 1
    W[1*2+1][col] = 1
    col += 1  # col=6
    
    # m7 = (a2-a4)(b3+b4)
    U[0*3+1][col] = 1; U[1*3+1][col] = -1
    V[1*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = -1
    col += 1  # col=7
    
    # Schoolbook for A[0:2,0:2] x B[0:2,2:5] -> C[0:2,2:5]
    # 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*5+(k+2)][col] = 1
                W[(k+2)*2+i][col] = 1
                col += 1
    # col = 7 + 12 = 19
    
    # Schoolbook for A[0:2,2] x B[2,0:5] -> C[0:2,0:5]
    # 2*1*5 = 10 terms
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    # col = 19 + 10 = 29
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
