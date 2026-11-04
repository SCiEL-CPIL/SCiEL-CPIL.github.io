"""Program for 2x3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal (or near-minimal) exact tensor decomposition for 2x3x6 matrix multiplication.
    
    Approach: Block decomposition
    - Split A (2x3) into A1 (2x1) and A2 (2x2)
    - Split B (3x6) into B1 (1x6) and B2 (2x6)
    - C = A1*B1 + A2*B2
    
    A1*B1: 2x1x6 multiplication, rank 12 (schoolbook, since it's just 2*6 scalar mults)
    A2*B2: 2x2x6 multiplication
      Split B2 into three 2x2 blocks: B2_0, B2_1, B2_2
      Each 2x2x2 block uses Strassen with rank 7
      Total for A2*B2: 3 * 7 = 21
    
    Total rank: 12 + 21 = 33
    """
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0.0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 33)
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1 has elements A[i][0] for i=0,1
    # B1 has elements B[0][k] for k=0..5
    # C[i][k] = A[i][0] * B[0][k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    # col is now 12
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2 elements: A[i][j] where i=0,1 and j=1,2
    # B2 elements: B[j][k] where j=1,2 and k=0..5
    # Split B2 into three 2x2 blocks along the k dimension
    
    for t in range(3):
        # B2 block t has columns 2t, 2t+1
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # A2 indices: (i, j) where i=0,1 and j=1,2
        # In U: row = i*m + j
        # B2_block indices: (j, k') where j=1,2 and k'=0,1
        # In V: row = j*p + (2t + k')
        # C indices: (i, k) where i=0,1 and k=2t, 2t+1
        # In W: row = k*n + i = (2t+k')*n + i
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1][col] = 1.0
        U[1 * m + 2][col] = 1.0
        V[1 * p + 2*t][col] = 1.0
        V[2 * p + 2*t + 1][col] = 1.0
        W[2*t * n + 0][col] = 1.0
        W[(2*t + 1) * n + 1][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1][col] = 1.0
        U[1 * m + 2][col] = 1.0
        V[1 * p + 2*t][col] = 1.0
        W[2*t * n + 1][col] = 1.0
        W[(2*t + 1) * n + 1][col] = -1.0
        col += 1
        
        # M3 = A00(B01-B11)
        U[0 * m + 1][col] = 1.0
        V[1 * p + 2*t + 1][col] = 1.0
        V[2 * p + 2*t + 1][col] = -1.0
        W[(2*t + 1) * n + 0][col] = 1.0
        W[(2*t + 1) * n + 1][col] = 1.0
        col += 1
        
        # M4 = A11(B10-B00)
        U[1 * m + 2][col] = 1.0
        V[2 * p + 2*t][col] = 1.0
        V[1 * p + 2*t][col] = -1.0
        W[2*t * n + 0][col] = 1.0
        W[2*t * n + 1][col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1][col] = 1.0
        U[0 * m + 2][col] = 1.0
        V[2 * p + 2*t + 1][col] = 1.0
        W[2*t * n + 0][col] = -1.0
        W[(2*t + 1) * n + 0][col] = 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1][col] = 1.0
        U[0 * m + 1][col] = -1.0
        V[1 * p + 2*t][col] = 1.0
        V[1 * p + 2*t + 1][col] = 1.0
        W[(2*t + 1) * n + 1][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2][col] = 1.0
        U[1 * m + 2][col] = -1.0
        V[2 * p + 2*t][col] = 1.0
        V[2 * p + 2*t + 1][col] = 1.0
        W[2*t * n + 0][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
