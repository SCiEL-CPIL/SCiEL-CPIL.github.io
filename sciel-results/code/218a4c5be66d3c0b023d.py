"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 26 decomposition: Strassen for 2x2 core, Schoolbook for border.
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2 core (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M3 = A00(B11-B00)
    U[0][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M4 = A11(B00-B10)
    U[4][col] = 1
    V[0][col] = 1; V[3][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # Schoolbook for border (indices involving 2)
    # C02 = A00B02 + A01B12 + A02B22
    U[0][col] = 1; V[2][col] = 1; W[2][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[2][col] = 1; col += 1
    U[2][col] = 1; V[8][col] = 1; W[2][col] = 1; col += 1
    
    # C12 = A10B02 + A11B12 + A12B22
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    # C20 = A20B00 + A21B10 + A22B20
    U[6][col] = 1; V[0][col] = 1; W[6][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    U[8][col] = 1; V[6][col] = 1; W[6][col] = 1; col += 1
    
    # C21 = A20B01 + A21B11 + A22B21
    U[6][col] = 1; V[1][col] = 1; W[7][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[7][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[7][col] = 1; col += 1
    
    # C22 = A20B02 + A21B12 + A22B22
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
