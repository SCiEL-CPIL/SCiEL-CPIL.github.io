"""Rewritten program for 3x3x7 matrix multiplication tensor rank minimization.

Approach: Block decomposition of the 3x3 @ 3x7 product into four block products,
using Strassen's 7-rank algorithm for the 2x2x2 sub-products and schoolbook for
the smaller rectangular sub-products.

Block structure:
  A (3x3) = [[A11 (2x2), A12 (2x1)],
             [A21 (1x2), A22 (1x1)]]
  B (3x7) = [[B1 (2x7)],
             [B2 (1x7)]]
  C (3x7) = [[C1 (2x7) = A11*B1 + A12*B2],
             [C2 (1x7) = A21*B1 + A22*B2]]

Rank contributions:
  A11*B1 (2x2x7): Split B1 into B1[0:2] (2x2), B1[2:4] (2x2), B1[4:6] (2x2), B1[6:7] (2x1)
    -> 3 * Strassen(2x2x2) + schoolbook(2x2x1) = 3*7 + 4 = 25
  A12*B2 (2x1x7): schoolbook = 14
  A21*B1 (1x2x7): schoolbook = 14
  A22*B2 (1x1x7): schoolbook = 7
  Total rank = 25 + 14 + 14 + 7 = 60
"""

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    rank = 60

    # U shape: (n*m, rank) = (9, 60)
    # V shape: (m*p, rank) = (21, 60)
    # W shape: (p*n, rank) = (21, 60)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0

    # --- Strassen 2x2x2 decomposition for A11 (A[0:2, 0:2]) * B_block (2 x 2) ---
    # Strassen's 7 multiplications for 2x2 matrix multiply:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # In our context, the "A" matrix for Strassen is A11 = A[0:2, 0:2]
    # The "B" matrix is a 2x2 block of B1 = B[0:2, 0:7]

    def strassen_contribution(a_row1, a_row2, b_col1, b_col2, out_col_offset):
        """
        Add Strassen contributions for A11 (rows a_row1, a_row2 of A; cols 0,1) 
        times B[0:2, b_col1:b_col2+1] to C[0:2, b_col1:b_col2+1].
        
        A11 is 2x2:
          A11[0,0] = A[a_row1, 0]
          A11[0,1] = A[a_row1, 1]
          A11[1,0] = A[a_row2, 0]
          A11[1,1] = A[a_row2, 1]
        
        B block is 2x2:
          B[0,0] = B[0, b_col1]
          B[0,1] = B[0, b_col1+1]
          B[1,0] = B[1, b_col1]
          B[1,1] = B[1, b_col1+1]
        
        C output block is 2x2:
          C[0,0] = C[a_row1, b_col1]   -> W index: b_col1 * n + a_row1
          C[0,1] = C[a_row1, b_col1+1] -> W index: (b_col1+1) * n + a_row1
          C[1,0] = C[a_row2, b_col1]   -> W index: b_col1 * n + a_row2
          C[1,1] = C[a_row2, b_col1+1] -> W index: (b_col1+1) * n + a_row2
        """
        global col
        nonlocal_col = col
        
        # A11 entries and their U indices
        # U index for A[i,j] is i*m + j = i*3 + j
        u_00 = a_row1 * m + 0  # A[a_row1, 0]
        u_01 = a_row1 * m + 1  # A[a_row1, 1]
        u_10 = a_row2 * m + 0  # A[a_row2, 0]
        u_11 = a_row2 * m + 1  # A[a_row2, 1]
        
        # B block entries and their V indices
        # V index for B[i,j] is i*p + j
        v_00 = 0 * p + b_col1      # B[0, b_col1]
        v_01 = 0 * p + (b_col1 + 1) # B[0, b_col1+1]
        v_10 = 1 * p + b_col1      # B[1, b_col1]
        v_11 = 1 * p + (b_col1 + 1) # B[1, b_col1+1]
        
        # C output entries and their W indices
        # W index for C[i,j] is j*n + i
        w_00 = b_col1 * n + a_row1
        w_01 = (b_col1 + 1) * n + a_row1
        w_10 = b_col1 * n + a_row2
        w_11 = (b_col1 + 1) * n + a_row2
        
        # Strassen M1 = (A11[0,0]+A11[1,1]) * (B[0,0]+B[1,1])
        # A part: A11[0,0] + A11[1,1]
        # B part: B[0,0] + B[1,1]
        # C contribution: C11 += M1, C22 += M1
        U[u_00][col] = 1.0
        U[u_11][col] = 1.0
        V[v_00][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = 1.0
        W[w_11][col] = 1.0
        col += 1
        
        # Strassen M2 = (A11[1,0]+A11[1,1]) * B[0,0]
        # C contribution: C21 += M2, C22 -= M2
        U[u_10][col] = 1.0
        U[u_11][col] = 1.0
        V[v_00][col] = 1.0
        W[w_10][col] = 1.0
        W[w_11][col] = -1.0
        col += 1
        
        # Strassen M3 = A11[0,0] * (B[0,1]-B[1,1])
        # C contribution: C12 += M3, C22 += M3
        U[u_00][col] = 1.0
        V[v_01][col] = 1.0
        V[v_11][col] = -1.0
        W[w_01][col] = 1.0
        W[w_11][col] = 1.0
        col += 1
        
        # Strassen M4 = A11[1,1] * (B[1,0]-B[0,0])
        # C contribution: C11 += M4, C21 += M4
        U[u_11][col] = 1.0
        V[v_10][col] = 1.0
        V[v_00][col] = -1.0
        W[w_00][col] = 1.0
        W[w_10][col] = 1.0
        col += 1
        
        # Strassen M5 = (A11[0,0]+A11[0,1]) * B[1,1]
        # C contribution: C11 -= M5, C12 += M5
        U[u_00][col] = 1.0
        U[u_01][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = -1.0
        W[w_01][col] = 1.0
        col += 1
        
        # Strassen M6 = (A11[1,0]-A11[0,0]) * (B[0,0]+B[0,1])
        # C contribution: C22 += M6
        U[u_10][col] = 1.0
        U[u_00][col] = -1.0
        V[v_00][col] = 1.0
        V[v_01][col] = 1.0
        W[w_11][col] = 1.0
        col += 1
        
        # Strassen M7 = (A11[0,1]-A11[1,1]) * (B[1,0]+B[1,1])
        # C contribution: C11 += M7
        U[u_01][col] = 1.0
        U[u_11][col] = -1.0
        V[v_10][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = 1.0
        col += 1

    # A11*B1: A11 = A[0:2, 0:2], B1 = B[0:2, 0:7]
    # Split into three 2x2x2 blocks (cols 0-1, 2-3, 4-5) and one 2x2x1 (col 6)
    
    # Strassen for B1[0:2, 0:2]
    strassen_contribution(0, 1, 0, 1, 0)
    # Strassen for B1[0:2, 2:4]
    strassen_contribution(0, 1, 2, 3, 0)
    # Strassen for B1[0:2, 4:6]
    strassen_contribution(0, 1, 4, 5, 0)
    
    # Schoolbook for A11 * B1[0:2, 6:7] (2x2x1)
    # A11[0,0]*B[0,6] -> C[0,6]
    # A11[0,1]*B[1,6] -> C[0,6]
    # A11[1,0]*B[0,6] -> C[1,6]
    # A11[1,1]*B[1,6] -> C[1,6]
    
    # A11[0,0] = A[0,0], V: B[0,6] = V[0*7+6]=V[6], W: C[0,6] = W[6*3+0]=W[18]
    U[0 * m + 0][col] = 1.0
    V[0 * p + 6][col] = 1.0
    W[6 * n + 0][col] = 1.0
    col += 1
    
    # A11[0,1] = A[0,1], V: B[1,6] = V[1*7+6]=V[13], W: C[0,6] = W[18]
    U[0 * m + 1][col] = 1.0
    V[1 * p + 6][col] = 1.0
    W[6 * n + 0][col] = 1.0
    col += 1
    
    # A11[1,0] = A[1,0], V: B[0,6] = V[6], W: C[1,6] = W[6*3+1]=W[19]
    U[1 * m + 0][col] = 1.0
    V[0 * p + 6][col] = 1.0
    W[6 * n + 1][col] = 1.0
    col += 1
    
    # A11[1,1] = A[1,1], V: B[1,6] = V[13], W: C[1,6] = W[19]
    U[1 * m + 1][col] = 1.0
    V[1 * p + 6][col] = 1.0
    W[6 * n + 1][col] = 1.0
    col += 1
    
    # --- A12*B2: A12 = A[0:2, 2] (2x1), B2 = B[2, 0:7] (1x7) ---
    # C contribution: C[0:2, 0:7] += A12 * B2
    # C[i, j] += A[i, 2] * B[2, j] for i in {0,1}, j in {0..6}
    # Schoolbook: 2*7 = 14 terms
    for i in range(2):
        for j in range(7):
            U[i * m + 2][col] = 1.0  # A[i, 2]
            V[2 * p + j][col] = 1.0  # B[2, j]
            W[j * n + i][col] = 1.0  # C[i, j]
            col += 1
    
    # --- A21*B1: A21 = A[2, 0:2] (1x2), B1 = B[0:2, 0:7] (2x7) ---
    # C contribution: C[2, 0:7] += A21 * B1
    # C[2, j] += A[2, 0]*B[0, j] + A[2, 1]*B[1, j] for j in {0..6}
    # Schoolbook: 1*2*7 = 14 terms
    for j in range(7):
        for k in range(2):
            U[2 * m + k][col] = 1.0  # A[2, k]
            V[k * p + j][col] = 1.0  # B[k, j]
            W[j * n + 2][col] = 1.0  # C[2, j]
            col += 1
    
    # --- A22*B2: A22 = A[2, 2] (1x1), B2 = B[2, 0:7] (1x7) ---
    # C contribution: C[2, 0:7] += A22 * B2
    # C[2, j] += A[2, 2] * B[2, j] for j in {0..6}
    # Schoolbook: 1*1*7 = 7 terms
    for j in range(7):
        U[2 * m + 2][col] = 1.0  # A[2, 2]
        V[2 * p + j][col] = 1.0  # B[2, j]
        W[j * n + 2][col] = 1.0  # C[2, j]
        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
