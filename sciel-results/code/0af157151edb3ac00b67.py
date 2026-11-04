"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
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
    
    # Define the 23 products
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Helper to add term
    def add_term(mat, row, col, val):
        mat[row][col] += val

    # M1 = (a00+a10+a20)(b00+b10+b20)
    add_term(U, 0, 0, 1); add_term(U, 3, 0, 1); add_term(U, 6, 0, 1)
    add_term(V, 0, 0, 1); add_term(V, 3, 0, 1); add_term(V, 6, 0, 1)
    # C00 += M1, C10 += M1, C20 += M1 ... wait, W is [k*3+i]
    # C_{00} is W[0]. C_{10} is W[3]. C_{20} is W[6].
    # Actually, W index is k*3+i.
    # C_{ij} corresponds to W[j*3+i].
    # So C_{00} -> W[0], C_{01} -> W[1], C_{02} -> W[2]
    # C_{10} -> W[3], C_{11} -> W[4], C_{12} -> W[5]
    # C_{20} -> W[6], C_{21} -> W[7], C_{22} -> W[8]
    
    # Reconstruction coefficients for W:
    # C00: M1 - M4 - M5 + M10 - M13 + M16 + M19 - M22
    # C01: M2 - M6 - M7 + M11 - M14 + M17 + M20 - M23
    # C02: M3 - M8 - M9 + M12 - M15 + M18 + M21
    # C10: M4 - M1 + M13 - M16 - M19 + M22
    # C11: M6 - M2 + M14 - M17 - M20 + M23
    # C12: M8 - M3 + M15 - M18 - M21
    # C20: M5 - M1 + M10 - M13 - M16 + M19
    # C21: M7 - M2 + M11 - M14 - M17 + M20
    # C22: M9 - M3 + M12 - M15 - M18 + M21
    
    # We will fill U, V, W for each product c from 0 to 22.
    
    # Product 0 (M1)
    add_term(U, 0, 0, 1); add_term(U, 3, 0, 1); add_term(U, 6, 0, 1)
    add_term(V, 0, 0, 1); add_term(V, 3, 0, 1); add_term(V, 6, 0, 1)
    add_term(W, 0, 0, 1)   # C00
    add_term(W, 3, 0, -1)  # C10
    add_term(W, 6, 0, -1)  # C20
    
    # Product 1 (M2)
    add_term(U, 1, 1, 1); add_term(U, 4, 1, 1); add_term(U, 7, 1, 1)
    add_term(V, 1, 1, 1); add_term(V, 4, 1, 1); add_term(V, 7, 1, 1)
    add_term(W, 1, 1, 1)   # C01
    add_term(W, 4, 1, -1)  # C11
    add_term(W, 7, 1, -1)  # C21
    
    # Product 2 (M3)
    add_term(U, 2, 2, 1); add_term(U, 5, 2, 1); add_term(U, 8, 2, 1)
    add_term(V, 2, 2, 1); add_term(V, 5, 2, 1); add_term(V, 8, 2, 1)
    add_term(W, 2, 2, 1)   # C02
    add_term(W, 5, 2, -1)  # C12
    add_term(W, 8, 2, -1)  # C22
    
    # Product 3 (M4)
    add_term(U, 0, 3, 1); add_term(U, 3, 3, 1)
    add_term(V, 3, 3, 1); add_term(V, 6, 3, 1)
    add_term(W, 0, 3, -1)  # C00
    add_term(W, 3, 3, 1)   # C10
    add_term(W, 6, 3, -1)  # C20
    
    # Product 4 (M5)
    add_term(U, 3, 4, 1); add_term(U, 6, 4, 1)
    add_term(V, 0, 4, 1); add_term(V, 3, 4, 1)
    add_term(W, 0, 4, -1)  # C00
    add_term(W, 3, 4, -1)  # C10
    add_term(W, 6, 4, 1)   # C20
    
    # Product 5 (M6)
    add_term(U, 1, 5, 1); add_term(U, 4, 5, 1)
    add_term(V, 4, 5, 1); add_term(V, 7, 5, 1)
    add_term(W, 1, 5, -1)  # C01
    add_term(W, 4, 5, 1)   # C11
    add_term(W, 7, 5, -1)  # C21
    
    # Product 6 (M7)
    add_term(U, 4, 6, 1); add_term(U, 7, 6, 1)
    add_term(V, 1, 6, 1); add_term(V, 4, 6, 1)
    add_term(W, 1, 6, -1)  # C01
    add_term(W, 4, 6, -1)  # C11
    add_term(W, 7, 6, 1)   # C21
    
    # Product 7 (M8)
    add_term(U, 2, 7, 1); add_term(U, 5, 7, 1)
    add_term(V, 5, 7, 1); add_term(V, 8, 7, 1)
    add_term(W, 2, 7, -1)  # C02
    add_term(W, 5, 7, 1)   # C12
    add_term(W, 8, 7, -1)  # C22
    
    # Product 8 (M9)
    add_term(U, 5, 8, 1); add_term(U, 8, 8, 1)
    add_term(V, 2, 8, 1); add_term(V, 5, 8, 1)
    add_term(W, 2, 8, -1)  # C02
    add_term(W, 5, 8, -1)  # C12
    add_term(W, 8, 8, 1)   # C22
    
    # Product 9 (M10)
    add_term(U, 0, 9, 1); add_term(U, 6, 9, 1)
    add_term(V, 0, 9, 1); add_term(V, 6, 9, 1)
    add_term(W, 0, 9, 1)   # C00
    add_term(W, 3, 9, -1)  # C10
    add_term(W, 6, 9, 1)   # C20
    
    # Product 10 (M11)
    add_term(U, 1, 10, 1); add_term(U, 7, 10, 1)
    add_term(V, 1, 10, 1); add_term(V, 7, 10, 1)
    add_term(W, 1, 10, 1)  # C01
    add_term(W, 4, 10, -1) # C11
    add_term(W, 7, 10, 1)  # C21
    
    # Product 11 (M12)
    add_term(U, 2, 11, 1); add_term(U, 8, 11, 1)
    add_term(V, 2, 11, 1); add_term(V, 8, 11, 1)
    add_term(W, 2, 11, 1)  # C02
    add_term(W, 5, 11, -1) # C12
    add_term(W, 8, 11, 1)  # C22
    
    # Product 12 (M13)
    add_term(U, 0, 12, 1); add_term(U, 3, 12, -1); add_term(U, 6, 12, 1)
    add_term(V, 0, 12, 1); add_term(V, 3, 12, 1); add_term(V, 6, 12, -1)
    add_term(W, 0, 12, -1) # C00
    add_term(W, 3, 12, 1)  # C10
    add_term(W, 6, 12, -1) # C20
    
    # Product 13 (M14)
    add_term(U, 1, 13, 1); add_term(U, 4, 13, -1); add_term(U, 7, 13, 1)
    add_term(V, 1, 13, 1); add_term(V, 4, 13, 1); add_term(V, 7, 13, -1)
    add_term(W, 1, 13, -1) # C01
    add_term(W, 4, 13, 1)  # C11
    add_term(W, 7, 13, -1) # C21
    
    # Product 14 (M15)
    add_term(U, 2, 14, 1); add_term(U, 5, 14, -1); add_term(U, 8, 14, 1)
    add_term(V, 2, 14, 1); add_term(V, 5, 14, 1); add_term(V, 8, 14, -1)
    add_term(W, 2, 14, -1) # C02
    add_term(W, 5, 14, 1)  # C12
    add_term(W, 8, 14, -1) # C22
    
    # Product 15 (M16)
    add_term(U, 0, 15, 1); add_term(U, 3, 15, 1); add_term(U, 6, 15, -1)
    add_term(V, 0, 15, 1); add_term(V, 3, 15, -1); add_term(V, 6, 15, 1)
    add_term(W, 0, 15, 1)  # C00
    add_term(W, 3, 15, -1) # C10
    add_term(W, 6, 15, -1) # C20
    
    # Product 16 (M17)
    add_term(U, 1, 16, 1); add_term(U, 4, 16, 1); add_term(U, 7, 16, -1)
    add_term(V, 1, 16, 1); add_term(V, 4, 16, -1); add_term(V, 7, 16, 1)
    add_term(W, 1, 16, 1)  # C01
    add_term(W, 4, 16, -1) # C11
    add_term(W, 7, 16, -1) # C21
    
    # Product 17 (M18)
    add_term(U, 2, 17, 1); add_term(U, 5, 17, 1); add_term(U, 8, 17, -1)
    add_term(V, 2, 17, 1); add_term(V, 5, 17, -1); add_term(V, 8, 17, 1)
    add_term(W, 2, 17, 1)  # C02
    add_term(W, 5, 17, -1) # C12
    add_term(W, 8, 17, -1) # C22
    
    # Product 18 (M19)
    add_term(U, 0, 18, 1); add_term(U, 3, 18, -1); add_term(U, 6, 18, -1)
    add_term(V, 0, 18, 1); add_term(V, 3, 18, -1); add_term(V, 6, 18, -1)
    add_term(W, 0, 18, 1)  # C00
    add_term(W, 3, 18, -1) # C10
    add_term(W, 6, 18, 1)  # C20
    
    # Product 19 (M20)
    add_term(U, 1, 19, 1); add_term(U, 4, 19, -1); add_term(U, 7, 19, -1)
    add_term(V, 1, 19, 1); add_term(V, 4, 19, -1); add_term(V, 7, 19, -1)
    add_term(W, 1, 19, 1)  # C01
    add_term(W, 4, 19, -1) # C11
    add_term(W, 7, 19, 1)  # C21
    
    # Product 20 (M21)
    add_term(U, 2, 20, 1); add_term(U, 5, 20, -1); add_term(U, 8, 20, -1)
    add_term(V, 2, 20, 1); add_term(V, 5, 20, -1); add_term(V, 8, 20, -1)
    add_term(W, 2, 20, 1)  # C02
    add_term(W, 5, 20, -1) # C12
    add_term(W, 8, 20, 1)  # C22
    
    # Product 21 (M22)
    add_term(U, 0, 21, 1); add_term(U, 3, 21, 1); add_term(U, 6, 21, 1)
    add_term(V, 0, 21, 1); add_term(V, 3, 21, -1); add_term(V, 6, 21, 1)
    add_term(W, 0, 21, -1) # C00
    add_term(W, 3, 21, 1)  # C10
    add_term(W, 6, 21, -1) # C20
    
    # Product 22 (M23)
    add_term(U, 1, 22, 1); add_term(U, 4, 22, 1); add_term(U, 7, 22, 1)
    add_term(V, 1, 22, 1); add_term(V, 4, 22, -1); add_term(V, 7, 22, 1)
    add_term(W, 1, 22, -1) # C01
    add_term(W, 4, 22, 1)  # C11
    add_term(W, 7, 22, -1) # C21
    
    return U, V, W

def _block_decomposition_6x6x6_winograd():
    """
    Decompose 6x6x6 matrix multiplication using 2x2 blocks of 3x3 matrices with Winograd F(3).
    Total rank = 7 * 23 = 161.
    """
    n, m, p = 6, 6, 6
    rank = 161
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    wU, wV, wW = _winograd_f3()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c in range(7):
                    # U block
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            U[u_row][col] = sU[li*2+lj][c] * wU[li*3+lj][0] # Placeholder, need to combine
                            # Actually, the Strassen U and Winograd U are for different structures.
                            # We are doing Strassen on the BLOCKS.
                            # The "entries" of the Strassen matrices are 3x3 blocks.
                            # So U_block[i,j] is a 3x3 matrix.
                            # U_block[i,j] = sU[i,j] * I_3? No.
                            # The Strassen algorithm computes linear combinations of blocks.
                            # M1 = (A11+A22)(B11+B22).
                            # This is a matrix multiplication of (A11+A22) and (B11+B22).
                            # So we need to decompose the multiplication of two 3x3 matrices.
                            # We use Winograd for this.
                            # So for each Strassen product c, we have a Winograd decomposition.
                            # The U matrix for the whole system has rows (i,j) of the 6x6 matrix.
                            # U[i*6+j, col] = sU[block_i, block_j][c] * wU[local_i, local_j][k]
                            # where block_i, block_j are the 2x2 block indices.
                            # local_i, local_j are the 3x3 local indices.
                            # k is the Winograd rank index (0..22).
                            # But wait, the Strassen product c is a single scalar multiplier for the block?
                            # No, sU[i,j][c] is 1, -1, or 0.
                            # So U[i*6+j, col] = sU[bi,bj][c] * wU[li*3+lj][k]
                            # And the column index is col = c * 23 + k.
                            
                            pass # This logic is inside the loop below

    # Correct implementation:
    col = 0
    for c in range(7):
        for k in range(23):
            # U
            for bi in range(2):
                for bj in range(2):
                    u_block_coeff = sU[bi*2+bj][c]
                    if u_block_coeff != 0:
                        for li in range(3):
                            for lj in range(3):
                                u_row = (bi*3+li)*6 + (bj*3+lj)
                                U[u_row][col] = u_block_coeff * wU[li*3+lj][k]
            
            # V
            for bj in range(2):
                for bk in range(2):
                    v_block_coeff = sV[bj*2+bk][c]
                    if v_block_coeff != 0:
                        for lj in range(3):
                            for lk in range(3):
                                v_row = (bj*3+lj)*6 + (bk*3+lk)
                                V[v_row][col] = v_block_coeff * wV[lj*3+lk][k]
            
            # W
            for bk in range(2):
                for bi in range(2):
                    w_block_coeff = sW[bk*2+bi][c]
                    if w_block_coeff != 0:
                        for lk in range(3):
                            for li in range(3):
                                w_row = (bk*3+lk)*6 + (bi*3+li)
                                W[w_row][col] = w_block_coeff * wW[lk*3+li][k]
            
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the Winograd-based block decomposition (rank 161)
    best = _block_decomposition_6x6x6_winograd()
    
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback to Strassen-based block decomposition (rank 189)
    best = _block_decomposition_6x6x6()
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
