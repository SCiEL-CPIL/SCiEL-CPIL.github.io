"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

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

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    The 6x6 matrix is divided into a 3x3 grid of 2x2 blocks.
    For each of the 27 block positions (bi, bj, bk) in {0,1,2}^3,
    we use Strassen (rank 7) for the 2x2 matrix multiplication.
    Total rank = 27 * 7 = 189.
    
    U row index: i = bi*2 + li, j = bj*2 + lj -> U row = i*6 + j
    V row index: j = bj*2 + lj, k = bk*2 + lk -> V row = j*6 + k  
    W row index: k = bk*2 + lk, i = bi*2 + li -> W row = k*6 + i
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

def _block_decomposition_2x2_of_3x3():
    """
    Alternative: 2x2 blocks of 3x3 matrices.
    Use Strassen (rank 7) for 2x2 block structure,
    Schoolbook (rank 27) for 3x3 matrix multiplication.
    Total rank = 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    # Schoolbook 3x3x3
    bU = [[0]*27 for _ in range(9)]
    bV = [[0]*27 for _ in range(9)]
    bW = [[0]*27 for _ in range(9)]
    col_idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bU[i*3+j][col_idx] = 1
                bV[j*3+k][col_idx] = 1
                bW[k*3+i][col_idx] = 1
                col_idx += 1
    
    col = 0
    for c in range(7):
        for d in range(27):
            for bi in range(2):
                for li in range(3):
                    for bj in range(2):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            val = sU[bi*2+bj][c] * bU[li*3+lj][d]
                            if val != 0:
                                U[u_row][col] = val
            
            for bj in range(2):
                for lj in range(3):
                    for bk in range(2):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            val = sV[bj*2+bk][c] * bV[lj*3+lk][d]
                            if val != 0:
                                V[v_row][col] = val
            
            for bk in range(2):
                for lk in range(3):
                    for bi in range(2):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            val = sW[bk*2+bi][c] * bW[lk*3+li][d]
                            if val != 0:
                                W[w_row][col] = val
            
            col += 1
    
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
    
    # Try 3x3 blocks of 2x2 with Strassen (rank 189)
    best = _block_decomposition_6x6x6()
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Try 2x2 blocks of 3x3 with Strassen+Schoolbook (rank 189)
    best = _block_decomposition_2x2_of_3x3()
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
