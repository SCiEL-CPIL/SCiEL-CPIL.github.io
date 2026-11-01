"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    # U shape: (n*m, rank) = (4, 11)
    # V shape: (m*p, rank) = (6, 11)
    # W shape: (p*n, rank) = (6, 11)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U (i, j): i in [0,1], j in [0,1] -> idx = i*m + j = i*2 + j
    # Indices for V (j, k): j in [0,1], k in [0,2] -> idx = j*p + k = j*3 + k
    # Indices for W (k, i): k in [0,2], i in [0,1] -> idx = k*n + i = k*2 + i
    
    # Term 1: z1 = (a11 + a22)(b11 + b22)
    # U: a11(0), a22(3)
    U[0][0] = 1
    U[3][0] = 1
    # V: b11(0), b22(4)
    V[0][0] = 1
    V[4][0] = 1
    # W: c11(idx 0), c22(idx 3)
    W[0][0] = 1
    W[3][0] = 1
    
    # Term 2: z2 = (a21 + a22)b11
    # U: a21(2), a22(3)
    U[2][1] = 1
    U[3][1] = 1
    # V: b11(0)
    V[0][1] = 1
    # W: c21(idx 1), c22(idx 3)
    W[1][1] = 1
    W[3][1] = 1
    
    # Term 3: z3 = a11(b12 - b22)
    # U: a11(0)
    U[0][2] = 1
    # V: b12(1), b22(4)
    V[1][2] = 1
    V[4][2] = -1
    # W: c12(idx 2), c22(idx 3)
    W[2][2] = 1
    W[3][2] = -1
    
    # Term 4: z4 = a22(b11 - b12)
    # U: a22(3)
    U[3][3] = 1
    # V: b11(0), b12(1)
    V[0][3] = 1
    V[1][3] = -1
    # W: c11(idx 0), c21(idx 1)
    W[0][3] = 1
    W[1][3] = 1
    
    # Term 5: z5 = (a11 + a12)b22
    # U: a11(0), a12(1)
    U[0][4] = 1
    U[1][4] = 1
    # V: b22(4)
    V[4][4] = 1
    # W: c11(idx 0), c12(idx 2)
    W[0][4] = -1
    W[2][4] = 1
    
    # Term 6: z6 = (a21 - a11)(b11 + b12)
    # U: a21(2), a11(0)
    U[2][5] = 1
    U[0][5] = -1
    # V: b11(0), b12(1)
    V[0][5] = 1
    V[1][5] = 1
    # W: c22(idx 3)
    W[3][5] = -1
    
    # Term 7: z7 = (a12 - a22)(b21 + b22)
    # U: a12(1), a22(3)
    U[1][6] = 1
    U[3][6] = -1
    # V: b21(3), b22(4)
    V[3][6] = 1
    V[4][6] = 1
    # W: c11(idx 0)
    W[0][6] = 1
    
    # Term 8: z8 = a11*b13
    # U: a11(0)
    U[0][7] = 1
    # V: b13(2)
    V[2][7] = 1
    # W: c13(idx 4)
    W[4][7] = 1
    
    # Term 9: z9 = a12*b23
    # U: a12(1)
    U[1][8] = 1
    # V: b23(5)
    V[5][8] = 1
    # W: c13(idx 4)
    W[4][8] = 1
    
    # Term 10: z10 = a21*b13
    # U: a21(2)
    U[2][9] = 1
    # V: b13(2)
    V[2][9] = 1
    # W: c23(idx 5)
    W[5][9] = 1
    
    # Term 11: z11 = a22*b23
    # U: a22(3)
    U[3][10] = 1
    # V: b23(5)
    V[5][10] = 1
    # W: c23(idx 5)
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
