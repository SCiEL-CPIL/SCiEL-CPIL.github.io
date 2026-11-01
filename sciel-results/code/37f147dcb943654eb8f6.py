"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # Rank 18: Strassen for cols 0-1 (7), Strassen for cols 2-3 (7), schoolbook for col 4 (4)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # (4, 18)
    V = [[0] * rank for _ in range(m * p)]  # (10, 18)
    W = [[0] * rank for _ in range(p * n)]  # (10, 18)
    
    def add_strassen_block(col0, col1, base):
        """Add 7 Strassen terms for 2x2 x 2x2 block with B columns col0, col1."""
        # M1: f=A00+A11, g=B[0,col0]+B[1,col1]
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        # M2: f=A10+A11, g=B[0,col0]
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        # M3: f=A00, g=B[0,col1]-B[1,col1]
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        # M4: f=A11, g=B[1,col0]-B[0,col0]
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        # M5: f=A00+A01, g=B[1,col1]
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        # M6: f=A10-A00, g=B[0,col0]+B[0,col1]
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        # M7: f=A01-A11, g=B[1,col0]+B[1,col1]
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    # Strassen block 1: columns 0, 1
    add_strassen_block(0, 1, 0)
    
    # Strassen block 2: columns 2, 3
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
