import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 69 decomposition
    rank = 69
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: C[1, k] for k in 0..5 (1-based: 1..6)
    # n=3, so row index i=0 (1-based 1)
    # m=5, j in 0..4
    # p=6, k in 0..5
    # For each k, we use 5 terms (one for each j)
    for k in range(p):
        for j in range(m):
            # U[i*m+j, col] = 1
            U[0 * m + j][col] = 1
            # V[j*p+k, col] = 1
            V[j * p + k][col] = 1
            # W[k*n+0, col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: C[2, k] and C[3, k] for k in 0..5
    # Rows i=1,2 (1-based 2,3)
    # Split m=5 into 2+2+1
    # M1: j=0,1
    # M2: j=2,3
    # M3: j=4
    
    # Strassen for 2x2x2
    # A is 2x2, B is 2x2
    # M1 = A11*B11 + A12*B21 ... etc
    # We need to map the Strassen multiplications to U, V, W
    
    # Strassen 2x2x2 rank 7
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We have 3 block columns K (k ranges)
    # K1: k=0,1
    # K2: k=2,3
    # K3: k=4,5
    
    for K in range(3):
        k_start = K * 2
        
        # Term 1: A[1:3, 0:2] * B[0:2, K] (2x2x2)
        # A rows 1,2 (0-based 1,2). B rows 0,1 (0-based 0,1).
        # This corresponds to M1 block.
        # We need to add 7 terms for Strassen.
        
        # Helper to add Strassen terms
        # a_rows: [r1, r2], a_cols: [c1, c2]
        # b_rows: [r1, r2], b_cols: [c1, c2]
        # c_rows: [r1, r2], c_cols: [c1, c2]
        
        def add_strassen(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2):
            nonlocal col
            # M1 = (A11+A22)(B11+B22)
            U[a_r1*m+a_c1][col] = 1
            U[a_r2*m+a_c2][col] = 1
            V[b_r1*p+b_c1][col] = 1
            V[b_r2*p+b_c2][col] = 1
            W[c_c1*n+c_r1][col] = 1
            W[c_c2*n+c_r1][col] = 1
            W[c_c1*n+c_r2][col] = 1
            W[c_c2*n+c_r2][col] = 1
            col += 1
            
            # M2 = (A21+A22)B11
            U[a_r2*m+a_c1][col] = 1
            U[a_r2*m+a_c2][col] = 1
            V[b_r1*p+b_c1][col] = 1
            W[c_c1*n+c_r2][col] = 1
            W[c_c2*n+c_r2][col] = 1
            col += 1
            
            # M3 = A11(B21-B22)
            U[a_r1*m+a_c1][col] = 1
            V[b_r2*p+b_c1][col] = 1
            V[b_r2*p+b_c2][col] = -1
            W[c_c1*n+c_r1][col] = 1
            W[c_c2*n+c_r1][col] = 1
            col += 1
            
            # M4 = A22(B12-B11)
            U[a_r2*m+a_c2][col] = 1
            V[b_r1*p+b_c2][col] = 1
            V[b_r1*p+b_c1][col] = -1
            W[c_c1*n+c_r1][col] = 1
            W[c_c2*n+c_r1][col] = 1
            W[c_c1*n+c_r2][col] = 1
            W[c_c2*n+c_r2][col] = 1
            col += 1
            
            # M5 = (A11+A12)B22
            U[a_r1*m+a_c1][col] = 1
            U[a_r1*m+a_c2][col] = 1
            V[b_r2*p+b_c2][col] = 1
            W[c_c1*n+c_r1][col] = -1
            W[c_c2*n+c_r1][col] = -1
            W[c_c1*n+c_r2][col] = -1
            W[c_c2*n+c_r2][col] = -1
            col += 1
            
            # M6 = (A21-A11)(B11+B12)
            U[a_r2*m+a_c1][col] = 1
            U[a_r1*m+a_c1][col] = -1
            V[b_r1*p+b_c1][col] = 1
            V[b_r1*p+b_c2][col] = 1
            W[c_c1*n+c_r2][col] = 1
            W[c_c2*n+c_r2][col] = 1
            col += 1
            
            # M7 = (A12-A22)(B21+B22)
            U[a_r1*m+a_c2][col] = 1
            U[a_r2*m+a_c2][col] = -1
            V[b_r2*p+b_c1][col] = 1
            V[b_r2*p+b_c2][col] = 1
            W[c_c1*n+c_r1][col] = 1
            W[c_c2*n+c_r1][col] = 1
            W[c_c1*n+c_r2][col] = 1
            W[c_c2*n+c_r2][col] = 1
            col += 1

        # Call for M1 block (j=0,1)
        # A rows 1,2 (0-based 1,2). Cols 0,1.
        # B rows 0,1 (0-based 0,1). Cols k_start, k_start+1.
        # C rows 1,2 (0-based 1,2). Cols k_start, k_start+1.
        add_strassen(1, 2, 0, 1, 0, 1, k_start, k_start+1, 1, 2, k_start, k_start+1)
        
        # Call for M2 block (j=2,3)
        # A rows 1,2. Cols 2,3.
        # B rows 2,3. Cols k_start, k_start+1.
        add_strassen(1, 2, 2, 3, 2, 3, k_start, k_start+1, 1, 2, k_start, k_start+1)
        
        # Term 3: A[1:3, 4] * B[4, K] (2x1x2)
        # Rank 4
        # A is 2x1 vector. B is 1x2 vector.
        # C11 = A1*B1, C12 = A1*B2, C21 = A2*B1, C22 = A2*B2
        # 4 terms
        # T1: A1*B1 -> C11
        U[1*m+4][col] = 1
        V[4*p+k_start][col] = 1
        W[k_start*n+1][col] = 1
        col += 1
        # T2: A1*B2 -> C12
        U[1*m+4][col] = 1
        V[4*p+(k_start+1)][col] = 1
        W[(k_start+1)*n+1][col] = 1
        col += 1
        # T3: A2*B1 -> C21
        U[2*m+4][col] = 1
        V[4*p+k_start][col] = 1
        W[k_start*n+2][col] = 1
        col += 1
        # T4: A2*B2 -> C22
        U[2*m+4][col] = 1
        V[4*p+(k_start+1)][col] = 1
        W[(k_start+1)*n+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
