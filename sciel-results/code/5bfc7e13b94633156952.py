"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Decompose 2x3x8 as (2x2x8) + (2x1x8)
    # 2x2x8: split into 4 blocks of 2x2x2, each with Strassen rank 7 -> total 28
    # 2x1x8: 16 scalar multiplications -> rank 16
    # Total rank: 44
    
    def build_strassen_2x2x2():
        """Build Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W matrices of shape (4,7) each.
        Indexing: U[i*2+j, r] corresponds to A[i,j], V[j*2+k, r] to B[j,k], W[k*2+i, r] to C[i,k]
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U: rows indexed by (i,j) in A, so row = i*2+j
        # V: rows indexed by (j,k) in B, so row = j*2+k  
        # W: rows indexed by (k,i) in C, so row = k*2+i
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        # C00 += M1, C11 += M1
        W[0][0] = 1  # C00 (k=0,i=0)
        W[3][0] = 1  # C11 (k=1,i=1)
        
        # M2 = (A10+A11)*B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        # C10 += M2, C11 -= M2
        W[2][1] = 1  # C10 (k=0,i=1)
        W[3][1] = -1  # C11 (k=1,i=1)
        
        # M3 = A00*(B01-B11)
        U[0][2] = 1  # A00
        V[1][2] = 1  # B01
        V[3][2] = -1  # B11
        # C01 += M3, C11 += M3
        W[1][2] = 1  # C01 (k=1,i=0)
        W[3][2] = 1  # C11 (k=1,i=1)
        
        # M4 = A11*(B10-B00)
        U[3][3] = 1  # A11
        V[2][3] = 1  # B10
        V[0][3] = -1  # B00
        # C00 += M4, C10 += M4
        W[0][3] = 1  # C00 (k=0,i=0)
        W[2][3] = 1  # C10 (k=0,i=1)
        
        # M5 = (A00+A01)*B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        # C00 -= M5, C01 += M5
        W[0][4] = -1  # C00 (k=0,i=0)
        W[1][4] = 1  # C01 (k=1,i=0)
        
        # M6 = (A10-A00)*(B00+B01)
        U[2][5] = 1  # A10
        U[0][5] = -1  # A00
        V[0][5] = 1  # B00
        V[1][5] = 1  # B01
        # C11 += M6
        W[3][5] = 1  # C11 (k=1,i=1)
        
        # M7 = (A01-A11)*(B10+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1  # A11
        V[2][6] = 1  # B10
        V[3][6] = 1  # B11
        # C00 += M7
        W[0][6] = 1  # C00 (k=0,i=0)
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p, timeout=10):
        """Verify that the tensor decomposition is correct."""
        start = time.time()
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        errors = 0
        for a in range(n*m):
            i, j = divmod(a, m)
            for b in range(m*p):
                j2, k = divmod(b, p)
                for c in range(p*n):
                    k2, i2 = divmod(c, n)
                    expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                    actual = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                    if abs(actual - expected) > 1e-9:
                        errors += 1
                    if time.time() - start > timeout:
                        return False
        return errors == 0
    
    # Build the rank-44 decomposition
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 24 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 16 rows
    
    # Part 1: 2x2x8 using Strassen (rank 28)
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x8) into B1 (2x8) and B2 (1x8)
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2x8, split into 4 blocks of 2x2x2
    # B1 is 2x8, so columns 0-1, 2-3, 4-5, 6-7 are four 2x2 blocks
    strassen_U, strassen_V, strassen_W = build_strassen_2x2x2()
    
    for block in range(4):
        base = block * 7
        # For each Strassen term r in 0..6:
        for r in range(7):
            col = base + r
            # U: A1[i,j] where i in {0,1}, j in {0,1} -> row = i*3 + j
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] += strassen_U[i * 2 + j][r]
            
            # V: B1[j,k] where j in {0,1}, k in {0,1} -> row = j*8 + (block*2 + k)
            for j in range(2):
                for k in range(2):
                    V[j * p + (block * 2 + k)][col] += strassen_V[j * 2 + k][r]
            
            # W: C[i,k] where i in {0,1}, k in {0,1} -> row = (block*2 + k) * 2 + i
            for i in range(2):
                for k in range(2):
                    W[(block * 2 + k) * n + i][col] += strassen_W[k * 2 + i][r]
    
    # Part 2: A2*B2: 2x1x8 (rank 16)
    # A2 is 2x1 (column j=2), B2 is 1x8 (row j=2)
    # C[i,k] += A2[i,2] * B2[2,k]
    base = 28
    for i in range(2):
        for k in range(8):
            col = base + i * 8 + k
            U[i * m + 2][col] = 1  # A2[i,2]
            V[2 * p + k][col] = 1  # B2[2,k]
            W[k * n + i][col] = 1  # C[i,k]
    
    # Verify the decomposition
    if not verify_decomposition(U, V, W, total_rank, n, m, p, timeout=20):
        # Fall back to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
