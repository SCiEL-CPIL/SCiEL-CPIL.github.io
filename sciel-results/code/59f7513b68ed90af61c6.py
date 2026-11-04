"""Matrix multiplication 2x3x10 - Strassen-based block decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x10 matrix multiplication.
    
    Strategy: Split the middle dimension m=3 into a 2+1 partition.
    - The 2x2 sub-block (rows 0-1 of A, cols 0-1 of B, producing 2x2 of C) 
      uses Strassen (rank 7 per 2x2x2 block).
    - The remaining 1x1 sub-block uses schoolbook.
    
    Actually, let me think more carefully about the structure.
    
    A is 2x3, B is 3x10, C = A*B is 2x10.
    
    We can split B's columns into two groups: first 2 columns and remaining 8 columns.
    Or split A's columns and B's rows.
    
    Let me think of it as: C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into j=0,1 (Strassen part) and j=2 (schoolbook part).
    
    For j=0,1: This is a 2x2 * 2x10 multiplication.
    We can further split the 10 columns into 5 blocks of 2 columns.
    For each 2-column block, we have a 2x2 * 2x2 multiplication, which can use Strassen (rank 7).
    
    So: 5 blocks * 7 (Strassen) + 2*10 (schoolbook for j=2) = 35 + 20 = 55
    
    Wait, let me reconsider. The schoolbook for j=2: for each i in {0,1} and k in {0..9}, 
    we need A[i,2]*B[2,k]. That's 2*10 = 20 terms.
    
    For the Strassen part (j=0,1): We have A[:,0:2] (2x2) times B[0:2,:] (2x10).
    Split B[0:2,:] into 5 blocks of 2 columns each.
    Each block is 2x2 * 2x2, using Strassen rank 7.
    Total: 5 * 7 = 35.
    
    Grand total rank: 35 + 20 = 55.
    
    Now let me construct the factor matrices.
    
    U: shape (n*m, rank) = (6, 55)
    V: shape (m*p, rank) = (30, 55)  
    W: shape (p*n, rank) = (20, 55)
    
    Indexing:
    U[i*m + j, c] where i in 0..1, j in 0..2
    V[j*p + k, c] where j in 0..2, k in 0..9
    W[k*n + i, c] where k in 0..9, i in 0..1
    
    Strassen for 2x2 * 2x2:
    Given A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    C = [[c11,c12],[c21,c22]]
    
    m1 = (a11+a22)*(b11+b22)
    m2 = (a21+a22)*b11
    m3 = a11*(b12-b22)
    m4 = a22*(b21-b11)
    m5 = (a11+a12)*b22
    m6 = (a21-a11)*(b11+b12)
    m7 = (a12-a22)*(b21+b22)
    
    c11 = m1+m4-m5+m7
    c12 = m3+m5
    c21 = m2+m4
    c22 = m1-m2+m3+m6
    
    For each 2-column block of B (columns 2*t and 2*t+1 for t=0..4):
    A_block = A[:, 0:2] = [[a11, a12], [a21, a22]] where aij = A[i, j]
    B_block = B[0:2, 2*t:2*t+2] = [[b11, b12], [b21, b22]] where bij = B[i, 2*t+j]
    
    C_block = C[:, 2*t:2*t+2] = [[c11, c12], [c21, c22]]
    
    For each Strassen term m_s (s=1..7):
    U entries: coefficients of A elements
    V entries: coefficients of B elements  
    W entries: coefficients of C elements
    
    Let me build this systematically.
    """
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen part: 5 blocks of 2x2 * 2x2
    # A[:, 0:2] is fixed, B[0:2, 2*t:2*t+2] varies with t
    
    # Strassen multipliers and their decompositions:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    
    # For each block t (0..4), for each Strassen term s (0..6):
    for t in range(5):
        # B block columns are 2*t and 2*t+1
        for s in range(7):
            # Determine U, V, W contributions for this Strassen term
            
            if s == 0:  # m1 = (a11+a22)*(b11+b22)
                # U: a11 coeff 1 at row 0*3+0=0, a22 coeff 1 at row 1*3+2=5
                U[0][col] = 1   # a11 = A[0,0]
                U[5][col] = 1   # a22 = A[1,2]
                # V: b11 coeff 1 at row 0*10+2*t, b22 coeff 1 at row 1*10+(2*t+1)
                V[0 * 10 + 2 * t][col] = 1      # b11 = B[0, 2*t]
                V[1 * 10 + (2 * t + 1)][col] = 1  # b22 = B[1, 2*t+1]
                # W: c11 coeff 1 at row (2*t)*2+0, c22 coeff 1 at row (2*t+1)*2+1
                W[(2 * t) * 2 + 0][col] = 1   # c11 = C[0, 2*t]
                W[(2 * t + 1) * 2 + 1][col] = 1  # c22 = C[1, 2*t+1]
                
            elif s == 1:  # m2 = (a21+a22)*b11
                U[3][col] = 1   # a21 = A[1,0]
                U[5][col] = 1   # a22 = A[1,2]
                V[0 * 10 + 2 * t][col] = 1      # b11
                W[(2 * t + 1) * 2 + 0][col] = 1   # c21 = C[1, 2*t]
                W[(2 * t + 1) * 2 + 1][col] = -1  # c22 = C[1, 2*t+1]
                
            elif s == 2:  # m3 = a11*(b12-b22)
                U[0][col] = 1   # a11
                V[0 * 10 + (2 * t + 1)][col] = 1   # b12 = B[0, 2*t+1]
                V[1 * 10 + (2 * t + 1)][col] = -1  # b22 = B[1, 2*t+1]
                W[(2 * t) * 2 + 1][col] = 1   # c12 = C[0, 2*t+1]
                W[(2 * t + 1) * 2 + 1][col] = 1  # c22 = C[1, 2*t+1]
                
            elif s == 3:  # m4 = a22*(b21-b11)
                U[5][col] = 1   # a22
                V[1 * 10 + 2 * t][col] = 1   # b21 = B[1, 2*t]
                V[0 * 10 + 2 * t][col] = -1  # b11 = B[0, 2*t]
                W[(2 * t) * 2 + 0][col] = 1   # c11 = C[0, 2*t]
                W[(2 * t + 1) * 2 + 0][col] = 1  # c21 = C[1, 2*t]
                
            elif s == 4:  # m5 = (a11+a12)*b22
                U[0][col] = 1   # a11
                U[1][col] = 1   # a12 = A[0,1]
                V[1 * 10 + (2 * t + 1)][col] = 1  # b22
                W[(2 * t) * 2 + 0][col] = -1  # c11 = C[0, 2*t]
                W[(2 * t) * 2 + 1][col] = 1   # c12 = C[0, 2*t+1]
                
            elif s == 5:  # m6 = (a21-a11)*(b11+b12)
                U[3][col] = 1   # a21
                U[0][col] = -1  # a11
                V[0 * 10 + 2 * t][col] = 1   # b11
                V[0 * 10 + (2 * t + 1)][col] = 1  # b12
                W[(2 * t + 1) * 2 + 1][col] = 1  # c22 = C[1, 2*t+1]
                
            elif s == 6:  # m7 = (a12-a22)*(b21+b22)
                U[1][col] = 1   # a12
                U[5][col] = -1  # a22
                V[1 * 10 + 2 * t][col] = 1   # b21
                V[1 * 10 + (2 * t + 1)][col] = 1  # b22
                W[(2 * t) * 2 + 0][col] = 1   # c11 = C[0, 2*t]
                
            col += 1
    
    # Schoolbook part: j=2, for each i in {0,1} and k in {0..9}
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][col] = 1   # A[i, 2]
            V[2 * 10 + k][col] = 1  # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
