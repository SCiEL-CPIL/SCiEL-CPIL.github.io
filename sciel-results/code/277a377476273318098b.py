"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51 decomposition
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(3):
        # Part 1: 2x2 x 2 (Strassen)
        # A indices: (0,0), (0,1), (1,0), (1,1) -> U rows 0,1,3,4
        # B indices: (0, 3t), (0, 3t+1), (1, 3t), (1, 3t+1) -> V rows 3t, 3t+1, 3+3t, 3+3t+1
        # W indices: (3t,0), (3t+1,0), (3t,1), (3t+1,1) -> W rows 6t, 6t+2, 6t+1, 6t+3
        
        # M1 = (a00+a11)(b00+b11)
        U[0][col] += 1; U[3][col] += 1
        V[3*t][col] += 1; V[3+3*t][col] += 1
        W[6*t][col] += 1; W[6*t+1][col] += 1; W[6*t+2][col] += 1; W[6*t+3][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[3][col] += 1; U[4][col] += 1
        V[3*t][col] += 1
        W[6*t+1][col] += 1; W[6*t+3][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[0][col] += 1
        V[3*t+1][col] += 1; V[3+3*t+1][col] -= 1
        W[6*t+2][col] += 1; W[6*t+3][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[4][col] += 1
        V[3+3*t][col] += 1; V[3*t][col] -= 1
        W[6*t][col] += 1; W[6*t+1][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[0][col] += 1; U[1][col] += 1
        V[3+3*t+1][col] += 1
        W[6*t][col] -= 1; W[6*t+2][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[3][col] += 1; U[0][col] -= 1
        V[3*t][col] += 1; V[3*t+1][col] += 1
        W[6*t+3][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][col] += 1; U[4][col] -= 1
        V[3+3*t][col] += 1; V[3+3*t+1][col] += 1
        W[6*t][col] += 1; W[6*t+3][col] += 1
        col += 1

        # Part 2: 2x2 x 1
        # T1 = a00 b0,3t+2
        U[0][col] += 1
        V[3*t+2][col] += 1
        W[6*t+4][col] += 1
        col += 1
        
        # T2 = a01 b1,3t+2
        U[1][col] += 1
        V[3+3*t+2][col] += 1
        W[6*t+4][col] += 1
        col += 1
        
        # T3 = a10 b0,3t+2
        U[3][col] += 1
        V[3*t+2][col] += 1
        W[6*t+5][col] += 1
        col += 1
        
        # T4 = a11 b1,3t+2
        U[4][col] += 1
        V[3+3*t+2][col] += 1
        W[6*t+5][col] += 1
        col += 1

        # Part 3: 2x1 x 3
        # U1 = a02 b2,3t
        U[2][col] += 1
        V[6+3*t][col] += 1
        W[6*t][col] += 1
        col += 1
        
        # U2 = a02 b2,3t+1
        U[2][col] += 1
        V[7+3*t][col] += 1
        W[6*t+2][col] += 1
        col += 1
        
        # U3 = a02 b2,3t+2
        U[2][col] += 1
        V[8+3*t][col] += 1
        W[6*t+4][col] += 1
        col += 1
        
        # U4 = a12 b2,3t
        U[5][col] += 1
        V[6+3*t][col] += 1
        W[6*t+1][col] += 1
        col += 1
        
        # U5 = a12 b2,3t+1
        U[5][col] += 1
        V[7+3*t][col] += 1
        W[6*t+3][col] += 1
        col += 1
        
        # U6 = a12 b2,3t+2
        U[5][col] += 1
        V[8+3*t][col] += 1
        W[6*t+5][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
