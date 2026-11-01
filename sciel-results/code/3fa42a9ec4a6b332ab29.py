"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # We split p=8 into 4 blocks of size 2.
    # For each block, we use a rank-11 decomposition of 2x3x2.
    # Total rank = 4 * 11 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    for b in range(4):
        # Block b corresponds to B columns 2b, 2b+1 and C columns 2b, 2b+1.
        # A is shared.
        
        # Strassen part for A1 (2x2) * B1 (2x2)
        # A1 indices: (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,3,4
        # B1 indices: (0, 2b), (0, 2b+1), (1, 2b), (1, 2b+1) -> V indices (0, 2b), (0, 2b+1), (1, 2b), (1, 2b+1)
        # C1 indices: (0, 2b), (0, 2b+1), (1, 2b), (1, 2b+1) -> W indices (2b, 0), (2b+1, 0), (2b, 1), (2b+1, 1)
        
        # p1 = (A00 + A11) * (B00 + B11)
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1)
        U[0][col] = 1; U[4][col] = 1
        V[0*8 + 2*b][col] = 1; V[1*8 + (2*b+1)][col] = 1
        W[2*b*2 + 0][col] = 1; W[(2*b+1)*2 + 1][col] = 1
        col += 1
        
        # p2 = (A10 + A11) * B00
        # U: A10(1), A11(1)
        # V: B00(1)
        # W: C10(1)
        U[3][col] = 1; U[4][col] = 1
        V[0*8 + 2*b][col] = 1
        W[2*b*2 + 1][col] = 1
        col += 1
        
        # p3 = A00 * (B01 - B11)
        # U: A00(1)
        # V: B01(1), B11(-1)
        # W: C01(1)
        U[0][col] = 1
        V[0*8 + (2*b+1)][col] = 1; V[1*8 + (2*b+1)][col] = -1
        W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # p4 = A11 * (B10 - B00)
        # U: A11(1)
        # V: B10(1), B00(-1)
        # W: C00(1), C10(1)
        U[4][col] = 1
        V[1*8 + 2*b][col] = 1; V[0*8 + 2*b][col] = -1
        W[2*b*2 + 0][col] = 1; W[2*b*2 + 1][col] = 1
        col += 1
        
        # p5 = (A00 + A01) * B11
        # U: A00(1), A01(1)
        # V: B11(1)
        # W: C00(-1), C01(1)
        U[0][col] = 1; U[1][col] = 1
        V[1*8 + (2*b+1)][col] = 1
        W[2*b*2 + 0][col] = -1; W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # p6 = (A10 - A00) * (B00 + B01)
        # U: A10(1), A00(-1)
        # V: B00(1), B01(1)
        # W: C11(1)
        U[3][col] = 1; U[0][col] = -1
        V[0*8 + 2*b][col] = 1; V[0*8 + (2*b+1)][col] = 1
        W[(2*b+1)*2 + 1][col] = 1
        col += 1
        
        # p7 = (A01 - A11) * (B10 + B11)
        # U: A01(1), A11(-1)
        # V: B10(1), B11(1)
        # W: C00(1)
        U[1][col] = 1; U[4][col] = -1
        V[1*8 + 2*b][col] = 1; V[1*8 + (2*b+1)][col] = 1
        W[2*b*2 + 0][col] = 1
        col += 1
        
        # Schoolbook part for A2 (2x1) * B2 (1x2)
        # A2 indices: (0,2), (1,2) -> U indices 2, 5
        # B2 indices: (2, 2b), (2, 2b+1) -> V indices (2, 2b), (2, 2b+1)
        
        # q1 = A02 * B20
        # U: A02(1)
        # V: B20(1)
        # W: C00(1)
        U[2][col] = 1
        V[2*8 + 2*b][col] = 1
        W[2*b*2 + 0][col] = 1
        col += 1
        
        # q2 = A02 * B21
        # U: A02(1)
        # V: B21(1)
        # W: C01(1)
        U[2][col] = 1
        V[2*8 + (2*b+1)][col] = 1
        W[(2*b+1)*2 + 0][col] = 1
        col += 1
        
        # q3 = A12 * B20
        # U: A12(1)
        # V: B20(1)
        # W: C10(1)
        U[5][col] = 1
        V[2*8 + 2*b][col] = 1
        W[2*b*2 + 1][col] = 1
        col += 1
        
        # q4 = A12 * B21
        # U: A12(1)
        # V: B21(1)
        # W: C11(1)
        U[5][col] = 1
        V[2*8 + (2*b+1)][col] = 1
        W[(2*b+1)*2 + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
