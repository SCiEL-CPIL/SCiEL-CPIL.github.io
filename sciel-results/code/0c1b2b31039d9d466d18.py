"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as 4x7 lists (flattened 2x2 matrices).
    Index: U[i*2+j][c], V[j*2+k][c], W[k*2+i][c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M1 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M2 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M3 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M4 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M5 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M6 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """Return the rank-23 Laderman decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as 9x23 lists (flattened 3x3 matrices).
    Index: U[i*3+j][c], V[j*3+k][c], W[k*3+i][c]
    """
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # k=0: (A00+A11+A22)(B00+B11+B22) -> C00, C11, C22
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    
    # k=1
    U[3][1] = 1; U[4][1] = 1
    V[3][1] = 1; V[4][1] = 1
    W[4][1] = 1
    
    # k=2
    U[6][2] = 1; U[7][2] = 1
    V[6][2] = 1; V[7][2] = 1
    W[8][2] = 1
    
    # k=3
    U[1][3] = 1; U[4][3] = 1
    V[1][3] = 1; V[4][3] = 1
    W[3][3] = 1
    
    # k=4
    U[2][4] = 1; U[5][4] = 1
    V[2][4] = 1; V[5][4] = 1
    W[6][4] = 1
    
    # k=5
    U[5][5] = 1; U[8][5] = 1
    V[5][5] = 1; V[8][5] = 1
    W[7][5] = 1
    
    # k=6
    U[7][6] = 1; U[8][6] = 1
    V[7][6] = 1; V[8][6] = 1
    W[5][6] = 1
    
    # k=7
    U[0][7] = 1; U[8][7] = 1
    V[8][7] = 1; V[0][7] = -1
    W[2][7] = 1; W[6][7] = 1
    
    # k=8
    U[3][8] = 1; U[8][8] = 1
    V[8][8] = 1; V[3][8] = -1
    W[5][8] = 1; W[7][8] = 1
    
    # k=9
    U[6][9] = 1; U[4][9] = 1
    V[4][9] = 1; V[6][9] = -1
    W[8][9] = 1; W[7][9] = -1
    
    # k=10
    U[1][10] = 1; U[8][10] = 1
    V[8][10] = 1; V[1][10] = -1
    W[2][10] = 1; W[6][10] = -1
    
    # k=11
    U[4][11] = 1; U[8][11] = 1
    V[8][11] = 1; V[4][11] = -1
    W[5][11] = 1; W[7][11] = -1
    
    # k=12
    U[2][12] = 1; U[4][12] = 1
    V[4][12] = 1; V[2][12] = -1
    W[3][12] = 1; W[6][12] = -1
    
    # k=13
    U[5][13] = 1; U[2][13] = 1
    V[2][13] = 1; V[5][13] = -1
    W[6][13] = 1; W[7][13] = -1
    
    # k=14
    U[7][14] = 1; U[1][14] = 1
    V[1][14] = 1; V[7][14] = -1
    W[3][14] = 1; W[5][14] = -1
    
    # k=15
    U[3][15] = 1; U[1][15] = 1
    V[1][15] = 1; V[3][15] = -1
    W[3][15] = 1; W[0][15] = -1
    
    # k=16
    U[6][16] = 1; U[2][16] = 1
    V[2][16] = 1; V[6][16] = -1
    W[6][16] = 1; W[2][16] = -1
    
    # k=17
    U[3][17] = 1; U[1][17] = 1
    V[1][17] = 1; V[3][17] = 1
    W[0][17] = 1; W[3][17] = -1
    
    # k=18
    U[6][18] = 1; U[2][18] = 1
    V[2][18] = 1; V[6][18] = 1
    W[2][18] = 1; W[6][18] = -1
    
    # k=19
    U[3][19] = 1; U[1][19] = 1
    V[1][19] = 1; V[3][19] = -1
    W[0][19] = -1; W[3][19] = 1
    
    # k=20
    U[6][20] = 1; U[2][20] = 1
    V[2][20] = 1; V[6][20] = -1
    W[2][20] = -1; W[6][20] = 1
    
    # k=21
    U[3][21] = 1; U[1][21] = 1
    V[1][21] = 1; V[3][21] = 1
    W[0][21] = -1; W[3][21] = 1
    
    # k=22
    U[6][22] = 1; U[2][22] = 1
    V[2][22] = 1; V[6][22] = 1
    W[2][22] = -1; W[6][22] = 1
    
    return U, V, W

def _kronecker_composition_6x6x6():
    """
    Compose Strassen 2x2 (rank 7) with Laderman 3x3 (rank 23) via Kronecker product.
    6x6 = 2x2 blocks of 3x3 matrices.
    Total rank = 7 * 23 = 161.
    """
    n, m, p = 6, 6, 6
    rank = 7 * 23  # 161
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    Us, Vs, Ws = _strassen_2x2()
    Ul, Vl, Wl = _laderman_3x3()
    
    for c in range(7):
        for d in range(23):
            col = c * 23 + d
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        # For each block position, fill in the 3x3 sub-blocks
                        for li in range(3):
                            for lj in range(3):
                                i = bi * 3 + li
                                j = bj * 3 + lj
                                u_val = Us[bi*2+bj][c] * Ul[li*3+lj][d]
                                if u_val != 0:
                                    U[i*6+j][col] = u_val
                        
                        for lj in range(3):
                            for lk in range(3):
                                j = bj * 3 + lj
                                k = bk * 3 + lk
                                v_val = Vs[bj*2+bk][c] * Vl[lj*3+lk][d]
                                if v_val != 0:
                                    V[j*6+k][col] = v_val
                        
                        for lk in range(3):
                            for li in range(3):
                                k = bk * 3 + lk
                                i = bi * 3 + li
                                w_val = Ws[bk*2+bi][c] * Wl[lk*3+li][d]
                                if w_val != 0:
                                    W[k*6+i][col] = w_val
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6_laderman():
    """
    Decompose 6x6x6 using 3x3 blocks with Laderman rank-23.
    Total rank = 8 * 23 = 184.
    """
    n, m, p = 6, 6, 6
    rank = 184
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _laderman_3x3()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(23):
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            U[u_row][col] = sU[li*3+lj][c]
                    
                    for lj in range(3):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            V[v_row][col] = sV[lj*3+lk][c]
                    
                    for lk in range(3):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            W[w_row][col] = sW[lk*3+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 using 3x3 blocks of 2x2 matrices (Strassen).
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

def _verify_decomposition(result, n, m, p, tolerance=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tolerance:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Kronecker composition of Strassen 2x2 (rank 7) and Laderman 3x3 (rank 23)
    # Total rank = 161
    if time.time() < deadline:
        best = _kronecker_composition_6x6x6()
        if _verify_decomposition(best, n, m, p):
            return best
    
    # Strategy 2: Laderman 3x3 blocks (rank 184)
    if time.time() < deadline:
        best = _block_decomposition_6x6x6_laderman()
        if _verify_decomposition(best, n, m, p):
            return best
    
    # Strategy 3: Strassen 2x2 blocks (rank 189)
    if time.time() < deadline:
        best = _block_decomposition_6x6x6()
        if _verify_decomposition(best, n, m, p):
            return best
    
    # Fallback: schoolbook (rank 216)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
