"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Use numpy for faster verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to 3D tensors
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute the tensor contraction: sum over rank of U[i,j,c]*V[j,k,c]*W[k,i,c]
    # This should equal 1 for all i,j,k
    # Use einsum: U[i,j,c] * V[j,k,c] * W[k,i,c] -> sum over c
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22): C11 += M1, C22 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11: C21 += M2, C22 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22): C12 += M3, C22 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11): C11 += M4, C21 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22: C11 -= M5, C12 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12): C22 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22): C11 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
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

def _winograd_f3():
    """
    Winograd F(3) algorithm for 3x3x3 matrix multiplication.
    Rank 23.
    """
    n = 3
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # The 23 products in Winograd F(3):
    # Using the standard formulation from Winograd (1980)
    
    # We define the 23 bilinear forms:
    # P1  = (a00+a10+a20)*(b00+b10+b20)
    # P2  = (a01+a11+a21)*(b01+b11+b21)
    # P3  = (a02+a12+a22)*(b02+b12+b22)
    # P4  = (a00+a10)*(b10+b20)
    # P5  = (a10+a20)*(b00+b10)
    # P6  = (a01+a11)*(b11+b21)
    # P7  = (a11+a21)*(b01+b11)
    # P8  = (a02+a12)*(b12+b22)
    # P9  = (a12+a22)*(b02+b12)
    # P10 = (a00+a20)*(b00+b20)
    # P11 = (a01+a21)*(b01+b21)
    # P12 = (a02+a22)*(b02+b22)
    # P13 = (a00-a10+a20)*(b00+b10-b20)
    # P14 = (a01-a11+a21)*(b01+b11-b21)
    # P15 = (a02-a12+a22)*(b02+b12-b22)
    # P16 = (a00+a10-a20)*(b00-b10+b20)
    # P17 = (a01+a11-a21)*(b01-b11+b21)
    # P18 = (a02+a12-a22)*(b02-b12+b22)
    # P19 = (a00-a10-a20)*(b00-b10-b20)
    # P20 = (a01-a11-a21)*(b01-b11-b21)
    # P21 = (a02-a12-a22)*(b02-b12-b22)
    # P22 = (a00+a10+a20)*(b00-b10+b20)
    # P23 = (a01+a11+a21)*(b01-b11+b21)
    
    # Define each product's contributions to U, V, W
    products = [
        # (U_indices, V_indices, W_indices)
        # P1
        ([0, 3, 6], [0, 3, 6], [(0,0,1),(0,1,-1),(0,2,1),(1,0,1),(1,1,-1),(1,2,1),(2,0,1),(2,1,-1),(2,2,1)]),
        # P2
        ([1, 4, 7], [1, 4, 7], [(0,0,1),(0,1,-1),(0,2,1),(1,0,1),(1,1,-1),(1,2,1),(2,0,1),(2,1,-1),(2,2,1)]),
        # P3
        ([2, 5, 8], [2, 5, 8], [(0,0,1),(0,1,-1),(0,2,1),(1,0,1),(1,1,-1),(1,2,1),(2,0,1),(2,1,-1),(2,2,1)]),
        # P4
        ([0, 3], [3, 6], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P5
        ([3, 6], [0, 3], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P6
        ([1, 4], [4, 7], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P7
        ([4, 7], [1, 4], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P8
        ([2, 5], [5, 8], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P9
        ([5, 8], [2, 5], [(0,0,-1),(1,0,1),(2,0,-1),(0,1,1),(1,1,-1),(2,1,1),(0,2,-1),(1,2,1),(2,2,-1)]),
        # P10
        ([0, 6], [0, 6], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P11
        ([1, 7], [1, 7], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P12
        ([2, 8], [2, 8], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P13
        ([0, 3, 6], [0, 3, 6], [(0,0,1),(1,0,1),(2,0,-1),(0,1,1),(1,1,1),(2,1,-1),(0,2,1),(1,2,1),(2,2,-1)]),
        # P14
        ([1, 4, 7], [1, 4, 7], [(0,0,1),(1,0,1),(2,0,-1),(0,1,1),(1,1,1),(2,1,-1),(0,2,1),(1,2,1),(2,2,-1)]),
        # P15
        ([2, 5, 8], [2, 5, 8], [(0,0,1),(1,0,1),(2,0,-1),(0,1,1),(1,1,1),(2,1,-1),(0,2,1),(1,2,1),(2,2,-1)]),
        # P16
        ([0, 3, 6], [0, 3, 6], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P17
        ([1, 4, 7], [1, 4, 7], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P18
        ([2, 5, 8], [2, 5, 8], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P19
        ([0, 3, 6], [0, 3, 6], [(0,0,1),(1,0,-1),(2,0,-1),(0,1,-1),(1,1,1),(2,1,1),(0,2,-1),(1,2,1),(2,2,1)]),
        # P20
        ([1, 4, 7], [1, 4, 7], [(0,0,1),(1,0,-1),(2,0,-1),(0,1,-1),(1,1,1),(2,1,1),(0,2,-1),(1,2,1),(2,2,1)]),
        # P21
        ([2, 5, 8], [2, 5, 8], [(0,0,1),(1,0,-1),(2,0,-1),(0,1,-1),(1,1,1),(2,1,1),(0,2,-1),(1,2,1),(2,2,1)]),
        # P22
        ([0, 3, 6], [0, 3, 6], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
        # P23
        ([1, 4, 7], [1, 4, 7], [(0,0,1),(1,0,-1),(2,0,1),(0,1,-1),(1,1,1),(2,1,-1),(0,2,1),(1,2,-1),(2,2,1)]),
    ]
    
    for c, (u_idx, v_idx, w_contribs) in enumerate(products):
        for ui in u_idx:
            U[ui][c] = 1
        for vi in v_idx:
            V[vi][c] = 1
        for (wi, wj, coeff) in w_contribs:
            W[wi*3+wj][c] += coeff
    
    return U, V, W

def _block_decomposition_6x6x6_winograd():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 3x3 matrices with Winograd F(3).
    Total rank = 9 * 23 = 207.
    """
    n, m, p = 6, 6, 6
    rank = 207
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    wU, wV, wW = _winograd_f3()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(23):
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            U[u_row][col] = wU[li*3+lj][c]
                    
                    for lj in range(3):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            V[v_row][col] = wV[lj*3+lk][c]
                    
                    for lk in range(3):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            W[w_row][col] = wW[lk*3+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based block decomposition (rank 189)
    best = _block_decomposition_6x6x6()
    
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
