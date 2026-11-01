"""Improved program for 2x3x6 matrix multiplication tensor decomposition.

Strategy:
- The 2x3x6 multiplication can be viewed as 3 independent 2x3x2 multiplications
  (one for each pair of columns in the 3x6 matrix).
- For each 2x3x2 multiplication, we partition the 2x3 matrix A into:
  * A_T: top-left 2x2 block (rows 0,1; cols 0,1)
  * A_R: right column (rows 0,1; col 2)
  and the 3x2 matrix B into:
  * B_T: top 2x2 block (rows 0,1; cols 0,1)
  * B_B: bottom row (row 2; cols 0,1)
  
  The product C = A*B is then:
  C = [A_T * B_T + A_R * B_B]
  
  Wait, that's not quite right. Let me think more carefully.
  
  A is 2x3: rows 0,1; cols 0,1,2
  B is 3x2: rows 0,1,2; cols 0,1
  
  C = A*B is 2x2:
  C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
  
  We can split the sum into k=0,1 and k=2:
  C[i,j] = (A[i,0]*B[0,j] + A[i,1]*B[1,j]) + A[i,2]*B[2,j]
  
  So C = A_T * B_T + A_R * B_B where:
  A_T is 2x2 (cols 0,1), B_T is 2x2 (rows 0,1)
  A_R is 2x1 (col 2), B_B is 1x2 (row 2)
  
  The first term is a 2x2x2 multiplication (rank 7 via Strassen)
  The second term is a 2x1x2 multiplication (rank 4 via schoolbook)
  
  Total per 2x3x2 block: 7 + 4 = 11 (vs schoolbook 12)
  For 3 blocks: 3 * 11 = 33 (vs schoolbook 36)
"""

import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We'll decompose into 3 independent 2x3x2 multiplications
    # For each block t (t=0,1,2), we handle columns 2t and 2t+1 of B
    # and the corresponding entries in C
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Index helper for W: W[k*n + i, col]
    # n=2, so W indices: k*2 + i
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * B11
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 - B11)
    #
    # C00 = M1 - M4 + M5 + M7  -> wait, let me double-check
    # Standard Strassen:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block t, the column offset in V is 3*(2t) + row_in_block
    # V row index: j*p + k where j is row of B (0,1,2), k is col of B (0..5)
    # For block t, B columns are 2t and 2t+1
    # B_T rows 0,1; B_B row 2
    # B_T col 0 -> B[0, 2t], B[1, 2t]
    # B_T col 1 -> B[0, 2t+1], B[1, 2t+1]
    # B_B col 0 -> B[2, 2t]
    # B_B col 1 -> B[2, 2t+1]
    
    # U row index: i*m + j where i is row of A (0,1), j is col of A (0,1,2)
    # A_T: j=0,1; A_R: j=2
    
    # W row index: k*n + i where k is col of C (0,1), i is row of C (0,1)
    # For block t, C columns are 2t and 2t+1
    
    col = 0
    
    for t in range(3):
        # Strassen part (7 products)
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 = U[0*3+0] = U[0], A11 = U[1*3+1] = U[4]
        # B00 = V[0*6+2t] = V[2t], B11 = V[1*6+2t+1] = V[6+2t+1]
        U[0][col] = 1
        U[4][col] = 1
        V[2*t][col] = 1
        V[6+2*t+1][col] = 1
        # W: C00 (k=2t, i=0) and C11 (k=2t+1, i=1)
        W[2*t*2+0][col] = 1   # C[0, 2t]
        W[(2*t+1)*2+1][col] = 1  # C[1, 2t+1]
        col += 1
        
        if time.time() > deadline:
            break
        
        # M2 = (A10 + A11) * B00
        # A10 = U[1*3+0] = U[3], A11 = U[4]
        # B00 = V[2t]
        U[3][col] = 1
        U[4][col] = 1
        V[2*t][col] = 1
        # W: C10 (k=2t, i=1)
        W[2*t*2+1][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # M3 = A00 * B11
        # A00 = U[0], B11 = V[6+2t+1]
        U[0][col] = 1
        V[6+2*t+1][col] = 1
        # W: C01 (k=2t+1, i=0) and C10 (k=2t, i=1)
        W[(2*t+1)*2+0][col] = 1  # C[0, 2t+1]
        W[2*t*2+1][col] = 1      # C[1, 2t]
        col += 1
        
        if time.time() > deadline:
            break
        
        # M4 = A11 * (B01 - B11)
        # A11 = U[4], B01 = V[6+2t+1], B11 = V[12+2t+1]
        U[4][col] = 1
        V[6+2*t+1][col] = 1
        V[12+2*t+1][col] = -1
        # W: C00 (k=2t, i=0), C01 (k=2t+1, i=0), C10 (k=2t, i=1)
        W[2*t*2+0][col] = -1   # C[0, 2t]
        W[(2*t+1)*2+0][col] = 1  # C[0, 2t+1]
        W[2*t*2+1][col] = 1     # C[1, 2t]
        col += 1
        
        if time.time() > deadline:
            break
        
        # M5 = (A00 + A01) * B11
        # A00 = U[0], A01 = U[1], B11 = V[12+2t+1]
        U[0][col] = 1
        U[1][col] = 1
        V[12+2*t+1][col] = 1
        # W: C00 (k=2t, i=0)
        W[2*t*2+0][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A10 = U[3], A00 = U[0], B00 = V[2t], B01 = V[6+2t+1]
        U[3][col] = 1
        U[0][col] = -1
        V[2*t][col] = 1
        V[6+2*t+1][col] = 1
        # W: C11 (k=2t+1, i=1)
        W[(2*t+1)*2+1][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # M7 = (A01 - A11) * (B10 - B11)
        # A01 = U[1], A11 = U[4], B10 = V[6+2t], B11 = V[12+2t+1]
        U[1][col] = 1
        U[4][col] = -1
        V[6+2*t][col] = 1
        V[12+2*t+1][col] = -1
        # W: C00 (k=2t, i=0)
        W[2*t*2+0][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # Outer product part: A_R * B_B (4 terms)
        # P1 = A02 * B20
        # A02 = U[0*3+2] = U[2], B20 = V[2*6+2t] = V[12+2t]
        U[2][col] = 1
        V[12+2*t][col] = 1
        # W: C00 (k=2t, i=0)
        W[2*t*2+0][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # P2 = A02 * B21
        # A02 = U[2], B21 = V[12+2t+1]
        U[2][col] = 1
        V[12+2*t+1][col] = 1
        # W: C01 (k=2t+1, i=0)
        W[(2*t+1)*2+0][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # P3 = A12 * B20
        # A12 = U[1*3+2] = U[5], B20 = V[12+2t]
        U[5][col] = 1
        V[12+2*t][col] = 1
        # W: C10 (k=2t, i=1)
        W[2*t*2+1][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
        
        # P4 = A12 * B21
        # A12 = U[5], B21 = V[12+2t+1]
        U[5][col] = 1
        V[12+2*t+1][col] = 1
        # W: C11 (k=2t+1, i=1)
        W[(2*t+1)*2+1][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}

