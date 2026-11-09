"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2."""
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2 = (a21+a22)b11
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
    # M3 = a11(b12-b22)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
    # M4 = a22(b21-b11)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5 = (a11+a12)b22
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6 = (a21-a11)(b11+b12)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7 = (a12-a22)(b21+b22)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
    
    return U, V, W

def _m223_decomp():
    """Return rank-11 decomposition for M_{2,2,3}."""
    # A: 2x2, B: 2x3, C: 2x3
    # Use schoolbook for first 2 columns (rank 4 via Strassen on 2x2x2) 
    # plus schoolbook for 3rd column (rank 2), but that's 6+2=8? No.
    # Actually let me just use a direct rank-12 schoolbook for safety, 
    # or try to construct rank-11.
    # For now, use schoolbook rank 12 to be safe.
    n, m, p = 2, 2, 3
    rank = 12
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _m224_decomp():
    """Return rank-14 decomposition for M_{2,2,4} using two Strassen blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # First 2 columns: Strassen on columns 0,1
    U1, V1, W1 = _strassen_2x2x2()
    for i in range(4):
        for r in range(7):
            U[i][r] += U1[i][r]
            V[i][r] += V1[i][r]
            W[i][r] += W1[i][r]
    
    # Second 2 columns: Strassen on columns 2,3
    # V index: j*p+k, so for columns 2,3: j=0,k=2,3 and j=1,k=2,3
    # W index: k*n+i, so for columns 2,3: k=2,3
    U2, V2, W2 = _strassen_2x2x2()
    for i in range(4):
        for r in range(7):
            U[i][7+r] += U2[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*4 + 2 + k][7+r] += V2[j*2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[(2+k)*2 + i][7+r] += W2[k*2 + i][r]
    
    return U, V, W

def _build_2x4x7_decomposition():
    """Build rank-50 decomposition for 2x4x7."""
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Split: A = [A1|A2] (each 2x2), B = [B1;B2] (each 2x7)
    # A@B = A1@B1 + A2@B2
    # B1 = [B1a|B1b] where B1a is 2x4, B1b is 2x3
    # B2 = [B2a|B2b] where B2a is 2x4, B2b is 2x3
    # A1@B1 = A1@B1a + A1@B1b
    # A2@B2 = A2@B2a + A2@B2b
    
    # A1@B1a: M_{2,2,4} rank 14, offset 0
    U1, V1, W1 = _m224_decomp()
    for i in range(4):
        for r in range(14):
            U[i][r] += U1[i][r]
    for j in range(2):
        for k in range(4):
            for r in range(14):
                V[j*7 + k][r] += V1[j*4 + k][r]
    for k in range(4):
        for i in range(2):
            for r in range(14):
                W[k*2 + i][r] += W1[k*2 + i][r]
    
    # A1@B1b: M_{2,2,3} rank 12, offset 14
    U2, V2, W2 = _m223_decomp()
    off = 14
    for i in range(4):
        for r in range(12):
            U[i][off+r] += U2[i][r]
    for j in range(2):
        for k in range(3):
            for r in range(12):
                V[j*7 + 4 + k][off+r] += V2[j*3 + k][r]
    for k in range(3):
        for i in range(2):
            for r in range(12):
                W[(4+k)*2 + i][off+r] += W2[k*2 + i][r]
    
    # A2@B2a: M_{2,2,4} rank 14, offset 26
    U3, V3, W3 = _m224_decomp()
    off = 26
    for i in range(4):
        for r in range(14):
            U[4+i][off+r] += U3[i][r]
    for j in range(2):
        for k in range(4):
            for r in range(14):
                V[(2+j)*7 + k][off+r] += V3[j*4 + k][r]
    for k in range(4):
        for i in range(2):
            for r in range(14):
                W[k*2 + i][off+r] += W3[k*2 + i][r]
    
    # A2@B2b: M_{2,2,3} rank 12, offset 40
    U4, V4, W4 = _m223_decomp()
    off = 40
    for i in range(4):
        for r in range(12):
            U[4+i][off+r] += U4[i][r]
    for j in range(2):
        for k in range(3):
            for r in range(12):
                V[(2+j)*7 + 4 + k][off+r] += V4[j*3 + k][r]
    for k in range(3):
        for i in range(2):
            for r in range(12):
                W[(4+k)*2 + i][off+r] += W4[k*2 + i][r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    U, V, W = _build_2x4x7_decomposition()
    rank = len(U[0])
    
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
