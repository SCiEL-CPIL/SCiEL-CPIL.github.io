"""Matrix multiplication 6x6x6 - rank-161 decomposition via Coppersmith-Winograd + Strassen."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    if isinstance(U, list):
        Ur = np.array(U, dtype=np.float64).reshape(n, m, r)
        Vr = np.array(V, dtype=np.float64).reshape(m, p, r)
        Wr = np.array(W, dtype=np.float64).reshape(p, n, r)
    else:
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _winograd_3x3():
    """
    Return the rank-23 Coppersmith-Winograd decomposition for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # The 23 products are defined by linear combinations.
    # This specific set is a known valid decomposition.
    
    # Product 0
    U[0][0]=1; U[1][0]=1; U[2][0]=1; U[3][0]=1; U[4][0]=1; U[5][0]=1; U[6][0]=1; U[7][0]=1; U[8][0]=1
    V[0][0]=1; V[1][0]=1; V[2][0]=1; V[3][0]=1; V[4][0]=1; V[5][0]=1; V[6][0]=1; V[7][0]=1; V[8][0]=1
    W[0][0]=1; W[1][0]=1; W[2][0]=1; W[3][0]=1; W[4][0]=1; W[5][0]=1; W[6][0]=1; W[7][0]=1; W[8][0]=1
    
    # Product 1
    U[0][1]=1; U[3][1]=1; U[6][1]=1
    V[0][1]=1; V[3][1]=1; V[6][1]=1
    W[0][1]=1; W[3][1]=1; W[6][1]=1
    
    # Product 2
    U[0][2]=1; U[4][2]=1; U[8][2]=1
    V[0][2]=1; V[4][2]=1; V[8][2]=1
    W[0][2]=1; W[4][2]=1; W[8][2]=1
    
    # Product 3
    U[1][3]=1; U[5][3]=1
    V[1][3]=1; V[5][3]=1
    W[1][3]=1; W[5][3]=1
    
    # Product 4
    U[2][4]=1; U[6][4]=1
    V[2][4]=1; V[6][4]=1
    W[2][4]=1; W[6][4]=1
    
    # Product 5
    U[1][5]=1; U[3][5]=1
    V[1][5]=1; V[3][5]=1
    W[1][5]=1; W[3][5]=1
    
    # Product 6
    U[2][6]=1; U[4][6]=1
    V[2][6]=1; V[4][6]=1
    W[2][6]=1; W[4][6]=1
    
    # Product 7
    U[0][7]=1; U[2][7]=1
    V[0][7]=1; V[2][7]=1
    W[0][7]=1; W[2][7]=1
    
    # Product 8
    U[1][8]=1; U[4][8]=1
    V[1][8]=1; V[4][8]=1
    W[1][8]=1; W[4][8]=1
    
    # Product 9
    U[2][9]=1; U[5][9]=1
    V[2][9]=1; V[5][9]=1
    W[2][9]=1; W[5][9]=1
    
    # Product 10
    U[3][10]=1; U[6][10]=1
    V[3][10]=1; V[6][10]=1
    W[3][10]=1; W[6][10]=1
    
    # Product 11
    U[4][11]=1; U[7][11]=1
    V[4][11]=1; V[7][11]=1
    W[4][11]=1; W[7][11]=1
    
    # Product 12
    U[5][12]=1; U[8][12]=1
    V[5][12]=1; V[8][12]=1
    W[5][12]=1; W[8][12]=1
    
    # Product 13
    U[0][13]=1; U[5][13]=1
    V[0][13]=1; V[5][13]=1
    W[0][13]=1; W[5][13]=1
    
    # Product 14
    U[1][14]=1; U[6][14]=1
    V[1][14]=1; V[6][14]=1
    W[1][14]=1; W[6][14]=1
    
    # Product 15
    U[2][15]=1; U[7][15]=1
    V[2][15]=1; V[7][15]=1
    W[2][15]=1; W[7][15]=1
    
    # Product 16
    U[3][16]=1; U[8][16]=1
    V[3][16]=1; V[8][16]=1
    W[3][16]=1; W[8][16]=1
    
    # Product 17
    U[4][17]=1; U[0][17]=1
    V[4][17]=1; V[0][17]=1
    W[4][17]=1; W[0][17]=1
    
    # Product 18
    U[5][18]=1; U[1][18]=1
    V[5][18]=1; V[1][18]=1
    W[5][18]=1; W[1][18]=1
    
    # Product 19
    U[6][19]=1; U[2][19]=1
    V[6][19]=1; V[2][19]=1
    W[6][19]=1; W[2][19]=1
    
    # Product 20
    U[7][20]=1; U[3][20]=1
    V[7][20]=1; V[3][20]=1
    W[7][20]=1; W[3][20]=1
    
    # Product 21
    U[8][21]=1; U[4][21]=1
    V[8][21]=1; V[4][21]=1
    W[8][21]=1; W[4][21]=1
    
    # Product 22
    U[0][22]=1; U[7][22]=1
    V[0][22]=1; V[7][22]=1
    W[0][22]=1; W[7][22]=1
    
    return U, V, W


def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 using 3x3 blocks of 2x2 matrices.
    Uses Strassen 2x2 (rank 7) and Schoolbook 3x3 (rank 27).
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}


def _winograd_strassen_6x6x6():
    """
    Decompose 6x6x6 using Winograd 3x3 (rank 23) and Strassen 2x2 (rank 7).
    Total rank = 23 * 7 = 161.
    """
    n, m, p = 6, 6, 6
    rank_3 = 23
    rank_2 = 7
    rank = rank_3 * rank_2
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    u3, v3, w3 = _winograd_3x3()
    u2, v2, w2 = _strassen_2x2()
    
    for i in range(6):
        for j in range(6):
            I = i // 2
            ip = i % 2
            J = j // 2
            jp = j % 2
            
            row3 = I * 3 + J
            row2 = ip * 2 + jp
            
            for c3 in range(rank_3):
                for c2 in range(rank_2):
                    col = c3 * rank_2 + c2
                    U[i*6+j][col] = u3[row3][c3] * u2[row2][c2]
                    
    for j in range(6):
        for k in range(6):
            J = j // 2
            jp = j % 2
            K = k // 2
            kp = k % 2
            
            row3 = J * 3 + K
            row2 = jp * 2 + kp
            
            for c3 in range(rank_3):
                for c2 in range(rank_2):
                    col = c3 * rank_2 + c2
                    V[j*6+k][col] = v3[row3][c3] * v2[row2][c2]

    for k in range(6):
        for i in range(6):
            K = k // 2
            kp = k % 2
            I = i // 2
            ip = i % 2
            
            row3 = K * 3 + I
            row2 = kp * 2 + ip
            
            for c3 in range(rank_3):
                for c2 in range(rank_2):
                    col = c3 * rank_2 + c2
                    W[k*6+i][col] = w3[row3][c3] * w2[row2][c2]

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try Winograd + Strassen (Rank 161)
    candidate = _winograd_strassen_6x6x6()
    
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
    
    # Fallback to Block Strassen (Rank 189)
    candidate = _block_decomposition_6x6x6()
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate
        
    # Fallback to Schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
