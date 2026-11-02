"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    For A (2x2), B (2x2), C = A*B (2x2):
    Index A by [i,j] with i,j in {0,1}
    Index B by [j,k] with j,k in {0,1}
    Index C by [k,i] with k,i in {0,1}
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U: rows indexed by (i,j) for A, so 4 rows; columns by the 7 Strassen terms
    # V: rows indexed by (j,k) for B, so 4 rows; columns by the 7 Strassen terms
    # W: rows indexed by (k,i) for C, so 4 rows; columns by the 7 Strassen terms
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W

def construct():
    """
    Block-Strassen decomposition for 6x6 matrix multiplication.
    
    Partition A, B, C into 3x3 blocks of 2x2 submatrices.
    Each block product C_ij = sum_k A_ik * B_kj uses Strassen's rank-7 decomposition.
    Total rank = 9 blocks * 3 inner products * 7 Strassen terms = 189.
    
    Indexing:
    A[i,j] where i in [0,6), j in [0,6)
    B[j,k] where j in [0,6), k in [0,6)
    C[k,i] where k in [0,6), i in [0,6)  (trace convention W[k*n+i, r])
    
    Block indices:
    A block (bi, bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    B block (bj, bk) covers rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    C block (bk, bi) covers rows 2*bk to 2*bk+1, cols 2*bi to 2*bi+1
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    block_size = 2
    n_blocks = n // block_size  # 3
    
    # Strassen decomposition for 2x2
    sU, sV, sW = _strassen_2x2()
    
    total_rank = n_blocks * n_blocks * n_blocks * 7  # 3*3*3*7 = 189
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for bi in range(n_blocks):  # A row block
        for bk in range(n_blocks):  # B col block / C row block
            for bj in range(n_blocks):  # A col block / B row block
                # C block (bk, bi) = A block (bi, bj) * B block (bj, bk)
                for s in range(7):  # Strassen term
                    # U: A element at block (bi, bj), local index sU_row
                    for si in range(4):
                        if sU[si][s] != 0:
                            # A local index si: row si//2, col si%2
                            a_row = block_size * bi + si // 2
                            a_col = block_size * bj + si % 2
                            U[a_row * m + a_col][col] = sU[si][s]
                    
                    # V: B element at block (bj, bk), local index sV_row
                    for sj in range(4):
                        if sV[sj][s] != 0:
                            # B local index sj: row sj//2, col sj%2
                            b_row = block_size * bj + sj // 2
                            b_col = block_size * bk + sj % 2
                            V[b_row * p + b_col][col] = sV[sj][s]
                    
                    # W: C element at block (bk, bi), local index sW_row
                    for sk in range(4):
                        if sW[sk][s] != 0:
                            # C local index sk: row sk//2, col sk%2
                            # C row index in full matrix: block_size * bk + sk//2
                            # C col index in full matrix: block_size * bi + sk%2
                            # W index: k*n + i where k = C row, i = C col
                            c_row = block_size * bk + sk // 2
                            c_col = block_size * bi + sk % 2
                            W[c_row * n + c_col][col] = sW[sk][s]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
