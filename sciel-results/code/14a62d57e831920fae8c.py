"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Decompose M_{2,4,6} as two independent M_{2,2,6} multiplications.
    # Further decompose each M_{2,2,6} as three independent M_{2,2,2} multiplications.
    # Use Strassen's algorithm (rank 7) for each M_{2,2,2}.
    # Total rank = 2 * 3 * 7 = 42.
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen coefficients for M_{2,2,2}:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll construct the factor matrices by iterating over:
    # - which half of A (top or bottom 2x2 block): h in {0, 1}
    # - which pair of columns in B: col_pair in {0, 1, 2}
    # - which Strassen multiplication: s in {0, 1, 2, 3, 4, 5, 6}
    
    rank = 2 * 3 * 7
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficient tables
    # For each Strassen multiplication s, we need:
    # - A coefficients: which elements of A_{i,j} (i,j in {0,1}) and with what sign
    # - B coefficients: which elements of B_{j,k} (j,k in {0,1}) and with what sign
    # - C output: which elements of C_{i,k} (i,k in {0,1}) and with what sign
    
    # Let's define the Strassen multiplications explicitly.
    # A is 2x2: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B is 2x2: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C is 2x2: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # M1 = (A11 + A22)(B11 + B22)
    # A part: A11 + A22
    # B part: B11 + B22
    # C part: +C11 -C21? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes to: C11 with +1, C22 with +1
    
    strassen_A = [
        {(0,0): 1, (1,1): 1},      # M1: A11 + A22
        {(1,0): 1, (1,1): 1},      # M2: A21 + A22
        {(0,0): 1},                 # M3: A11
        {(1,1): 1},                 # M4: A22
        {(0,0): 1, (0,1): 1},      # M5: A11 + A12
        {(1,0): 1, (0,0): -1},     # M6: A21 - A11
        {(0,1): 1, (1,1): -1},     # M7: A12 - A22
    ]
    
    strassen_B = [
        {(0,0): 1, (1,1): 1},      # M1: B11 + B22
        {(0,0): 1},                 # M2: B11
        {(0,1): 1, (1,1): -1},     # M3: B12 - B22
        {(1,0): 1, (0,0): -1},     # M4: B21 - B11
        {(1,1): 1},                 # M5: B22
        {(0,0): 1, (0,1): 1},      # M6: B11 + B12
        {(1,0): 1, (1,1): 1},      # M7: B21 + B22
    ]
    
    strassen_C = [
        {(0,0): 1, (1,1): 1},      # M1: +C11, +C22
        {(1,0): 1},                 # M2: +C21
        {(0,1): 1},                 # M3: +C12
        {(0,0): 1, (1,0): 1},      # M4: +C11, +C21
        {(0,0): -1, (0,1): 1},     # M5: -C11, +C12
        {(1,1): 1},                 # M6: +C22
        {(0,0): 1},                 # M7: +C11
    ]
    
    col = 0
    for h in range(2):  # top or bottom half of A
        for cp in range(3):  # which pair of columns in B
            for s in range(7):  # which Strassen multiplication
                # Fill U: A elements
                # A is 2x4. The h-th 2x2 block has rows h*2 to h*2+1, columns 2*(h) to... 
                # Wait, A is 2x4. We split A into two 2x2 blocks: A[0:2, 0:2] and A[0:2, 2:4].
                # So h=0 uses columns 0,1 and h=1 uses columns 2,3.
                for (ai, aj), coeff in strassen_A[s].items():
                    # ai is row in the 2x2 block (0 or 1), aj is col in 2x2 block (0 or 1)
                    # Actual row in A: ai (since A has only 2 rows)
                    # Actual col in A: 2*h + aj
                    row_U = ai  # 0 or 1
                    col_U = 2*h + aj  # 0,1,2,3
                    U[row_U][col] += coeff
                
                # Fill V: B elements
                # B is 4x6. We split B into 2x2 blocks.
                # For the h-th A block, we use the h-th row block of B: rows 2*h to 2*h+1.
                # For the cp-th column pair, we use columns 2*cp to 2*cp+1.
                for (bj, bk), coeff in strassen_B[s].items():
                    # bj is row in 2x2 block (0 or 1), bk is col in 2x2 block (0 or 1)
                    # Actual row in B: 2*h + bj
                    # Actual col in B: 2*cp + bk
                    row_V = 2*h + bj
                    col_V = 2*cp + bk
                    V[row_V][col] += coeff
                
                # Fill W: C elements
                # C is 6x2 (following W[k*n+i, r] convention, so W is p*n x rank = 12 x rank)
                # Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
                # And the shape of W is (p*n, rank).
                # So W is 12 x rank.
                # C is the output of matrix multiplication: A (2x4) * B (4x6) = C (2x6).
                # The convention W[k*n+i, r] means that for output element C[i, k] (where i is row of C, k is col of C),
                # it's stored at W[k*n + i, r].
                # So C[i, k] maps to W[k*2 + i, r].
                
                # For the h-th A block and cp-th B column pair, the output is a 2x2 block of C.
                # The rows of C are 0 and 1 (since C has 2 rows).
                # The columns of C for this block are 2*cp and 2*cp+1.
                
                for (ci, ck), coeff in strassen_C[s].items():
                    # ci is row in 2x2 output block (0 or 1), ck is col in 2x2 output block (0 or 1)
                    # Actual row in C: ci
                    # Actual col in C: 2*cp + ck
                    # W index: (2*cp + ck) * n + ci = (2*cp + ck) * 2 + ci
                    row_W = (2*cp + ck) * 2 + ci
                    W[row_W][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
