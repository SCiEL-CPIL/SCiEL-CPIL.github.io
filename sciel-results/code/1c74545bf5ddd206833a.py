"""5x5x5 matrix multiplication tensor decomposition using 2+3 block structure.
Strassen for 2x2x2 (rank 7) + rank-24 for 3x3x3 + schoolbook cross terms.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition. Returns (U,V,W) each 4x7.
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # W[k*2+i]: C00=W[0], C01=W[2], C10=W[1], C11=W[3]
    # A: A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B: B00=V[0], B01=V[1], B10=V[2], B11=V[3]
    
    # M1: (A00+A11)*(B00+B11) -> C00+1, C11+1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2: (A10+A11)*B00 -> C10+1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1
    # M3: A00*(B01-B11) -> C01+1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1
    # M4: A11*(B10-B00) -> C00+1, C11+1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[3][3]=1
    # M5: (A00+A01)*B11 -> C00-1, C01+1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6: (A10-A00)*(B00+B01) -> C11+1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7: (A01-A11)*(B10+B11) -> C00+1, C10-1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[1][6]=-1
    return U, V, W


def _rank24_3x3x3():
    """Rank-24 3x3x3 decomposition via 2+1 block with Strassen on 2x2 sub-block.
    Returns (U,V,W) each 9x24. U[i*3+j], V[j*3+k], W[k*3+i].
    """
    U = [[0]*24 for _ in range(9)]
    V = [[0]*24 for _ in range(9)]
    W = [[0]*24 for _ in range(9)]
    col = 0
    # A22 = A[0:2,0:2]: U indices 0,1,3,4
    # A21 = A[0:2,2]: U indices 2,5
    # A12 = A[2,0:2]: U indices 6,7
    # A11 = A[2,2]: U index 8
    # B22 = B[0:2,0:2]: V indices 0,1,3,4
    # B21 = B[0:2,2]: V indices 2,5
    # B12 = B[2,0:2]: V indices 6,7
    # B11 = B[2,2]: V index 8
    # C22 = C[0:2,0:2]: W indices 0,3,1,4 (W[k*3+i])
    # C21 = C[0:2,2]: W indices 6,7
    # C12 = C[2,0:2]: W indices 2,5
    # C11 = C[2,2]: W index 8
    
    # Strassen for A22*B22 -> C22
    sU, sV, sW = _strassen_2x2x2()
    # Map: sU[i*2+j] -> U[i*3+j], sV[j*2+k] -> V[j*3+k], sW[k*2+i] -> W[k*3+i]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i*3+j][col+c] += sU[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*3+k][col+c] += sV[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*3+i][col+c] += sW[k*2+i][c]
    col += 7
    
    # A21*B12 -> C22: A[0:2,2]*B[2,0:2] -> C[0:2,0:2]
    for i in range(2):
        for k in range(2):
            U[2+i][col] = 1
            V[6+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    
    # A22*B21 -> C21: A[0:2,0:2]*B[0:2,2] -> C[0:2,2]
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[6+i][col] = 1
            col += 1
    
    # A12*B22 -> C12: A[2,0:2]*B[0:2,0:2] -> C[2,0:2]
    for k in range(2):
        for j in range(2):
            U[6+j][col] = 1
            V[j*3+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    # A11*B12 -> C12: A[2,2]*B[2,0:2] -> C[2,0:2]
    for k in range(2):
        U[8][col] = 1
        V[6+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
    
    # A12*B21 -> C11: A[2,0:2]*B[0:2,2] -> C[2,2]
    for j in range(2):
        U[6+j][col] = 1
        V[j*3+2][col] = 1
        W[8][col] = 1
        col += 1
    
    # A11*B11 -> C11
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    return U, V, W


def _verify(U, V, W, n=5, m=5, p=5):
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        u = Ua[:, c].reshape(n, m)
        v = Va[:, c].reshape(m, p)
        w = Wa[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                tensor[i, j, :] += u[i, j] * (v[j, :] * w[:, i])
    return np.allclose(tensor, 1.0, atol=1e-9)


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS
    
    # Build 2+3 block decomposition
    # Total rank: 7 + 24 + 18 + 12 = 61? Let me recount cross terms.
    # A[0:2,2:5]*B[2:5,2:5] -> C[0:2,2:5]: 2*3*3 = 18
    # A[2:5,0:2]*B[0:2,0:2] -> C[2:5,0:2]: 3*2*2 = 12
    # Total: 7 + 24 + 18 + 12 = 61
    total_rank = 7 + 24 + 18 + 12
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    col = 0
    
    # Strassen 2x2x2: A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    sU, sV, sW = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i*5+j][col+c] += sU[i*2+j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j*5+k][col+c] += sV[j*2+k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k*5+i][col+c] += sW[k*2+i][c]
    col += 7
    
    # Rank-24 3x3x3: A[2:5,2:5]*B[2:5,2:5]->C[2:5,2:5]
    tU, tV, tW = _rank24_3x3x3()
    for i in range(3):
        for j in range(3):
            for c in range(24):
                U[(2+i)*5+(2+j)][col+c] += tU[i*3+j][c]
    for j in range(3):
        for k in range(3):
            for c in range(24):
                V[(2+j)*5+(2+k)][col+c] += tV[j*3+k][c]
    for k in range(3):
        for i in range(3):
            for c in range(24):
                W[(2+k)*5+(2+i)][col+c] += tW[k*3+i][c]
    col += 24
    
    # Cross term 1: A[0:2,2:5]*B[2:5,2:5]->C[0:2,2:5] (2x3x3=18)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*5+(2+j)][col] = 1
                V[(2+j)*5+(2+k)][col] = 1
                W[(2+k)*5+i][col] = 1
                col += 1
    
    # Cross term 2: A[2:5,0:2]*B[0:2,0:2]->C[2:5,0:2] (3x2x2=12)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[(2+i)*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+(2+i)][col] = 1
                col += 1
    
    # Verify
    if time.time() - t_start < 100:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = 125
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    column = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U[i*5+j][column] = 1
                V[j*5+k][column] = 1
                W[k*5+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
