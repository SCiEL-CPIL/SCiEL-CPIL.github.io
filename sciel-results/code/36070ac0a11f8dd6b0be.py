"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid decomposition:
    # 1. Strassen for the 2x2 block of A (cols 0,1) and B (rows 0,1) -> 2x2 output block
    # 2. Schoolbook for A (cols 0,1) and B (row 2) -> 2x1 output block
    # 3. Schoolbook for A (col 2) and B (rows 0,1,2) -> 2x3 output block
    # Wait, the output is 2x3.
    # Let's define the terms carefully.
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    # A_strassen = [[A00, A01], [A10, A11]]
    # B_strassen = [[B00, B01, B02], [B10, B11, B12]]
    # This is not square. 
    # Let's stick to the previous derivation:
    # A12 = A[:, 0:2] (2x2)
    # B12_L = B[0:2, 0:2] (2x2)
    # B12_R = B[0:2, 2:3] (2x1)
    # A3 = A[:, 2:3] (2x1)
    # B3 = B[2:3, :] (1x3)
    
    # 1. A12 * B12_L (2x2x2) Strassen
    # Inputs: A00, A01, A10, A11 ; B00, B01, B10, B11
    # Indices in U: A00=0, A01=1, A10=2, A11=3
    # Indices in V: B00=0, B01=3, B10=6, B11=9 (m=3, p=3 -> V index = j*p + k)
    # V indices:
    # B00: 0*3+0=0
    # B01: 0*3+1=1
    # B10: 1*3+0=3
    # B11: 1*3+1=4
    # B02: 0*3+2=2
    # B12: 1*3+2=5
    # B20: 2*3+0=6
    # B21: 2*3+1=7
    # B22: 2*3+2=8
    
    # Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Output of Strassen part is 2x2 (cols 0,1 of the 2x3 output)
    # W indices: k*n + i. n=2.
    # W00: 0*2+0=0
    # W01: 0*2+1=1
    # W10: 1*2+0=2
    # W11: 1*2+1=3
    
    # P1 contributes to all 4 outputs
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # P2 = (A10+A11)B00 -> contributes to W10, W11
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # P3 = A00(B01-B11) -> contributes to W01, W11
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # P4 = A11(B10-B00) -> contributes to W00, W01
    U[3][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5 = (A00+A01)B11 -> contributes to W00, W10
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B00+B01) -> contributes to W10, W11
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11) -> contributes to W00, W10
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # 2. A12 * B12_R (2x2x1) Schoolbook
    # A12 = [[A00, A01], [A10, A11]]
    # B12_R = [[B02], [B12]]
    # Output is 2x1 (col 2 of the 2x3 output)
    # W indices for col 2:
    # W02: 2*2+0=4
    # W12: 2*2+1=5
    
    # Q1 = A00 * B02
    U[0][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    col += 1
    
    # Q2 = A01 * B02
    U[1][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    col += 1
    
    # Q3 = A10 * B12
    U[2][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # Q4 = A11 * B12
    U[3][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    # 3. A3 * B3 (2x1x3) Schoolbook
    # A3 = [[A02], [A12]] -> U indices 4, 5
    # B3 = [[B20, B21, B22]] -> V indices 6, 7, 8
    # Output is 2x3 (all cols)
    # W indices: 0..5
    
    # R1 = A02 * B20
    U[4][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # R2 = A02 * B21
    U[4][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    
    # R3 = A02 * B22
    U[4][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # R4 = A12 * B20
    U[5][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    col += 1
    
    # R5 = A12 * B21
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # R6 = A12 * B22
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
