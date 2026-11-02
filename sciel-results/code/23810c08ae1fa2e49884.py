import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # The rank of the 2x2x4 multiplication tensor is 14.
    # This is achieved by decomposing it into two independent 2x2x2 multiplications.
    # Each 2x2x2 multiplication has a minimal rank of 7 (Strassen's algorithm).
    # Total rank = 7 + 7 = 14.
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (n*m = 4):
    # 0: a00, 1: a01, 2: a10, 3: a11
    
    # Indices for B (m*p = 8):
    # 0: b00, 1: b01, 2: b02, 3: b03
    # 4: b10, 5: b11, 6: b12, 7: b13
    
    # Indices for W (p*n = 8), corresponding to C (2x4):
    # Convention: W[k*n+i, r] corresponds to C[i, k]
    # k is column index of C (0..3), i is row index of C (0..1)
    # W index = k*n + i = k*2 + i
    # 0: c00 (k=0, i=0)
    # 1: c10 (k=0, i=1)
    # 2: c01 (k=1, i=0)
    # 3: c11 (k=1, i=1)
    # 4: c02 (k=2, i=0)
    # 5: c12 (k=2, i=1)
    # 6: c03 (k=3, i=0)
    # 7: c13 (k=3, i=1)

    # Block 1: Multiplying A by B[:, 0:2] to get C[:, 0:2]
    # This is a standard 2x2 x 2x2 multiplication.
    # We use Strassen's algorithm.
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10 + a11) * b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a00 * (b01 - b11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a11 * (b10 - b00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a00 + a01) * b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a10 - a00) * (b00 + b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01 - a11) * (b10 + b11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Block 2: Multiplying A by B[:, 2:4] to get C[:, 2:4]
    # This is another standard 2x2 x 2x2 multiplication.
    # A is the same. B columns are 2,3. C columns are 2,3.
    
    # N1 = (a00 + a11) * (b02 + b12)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[6][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (a10 + a11) * b02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = a00 * (b03 - b13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = a11 * (b12 - b02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (a00 + a01) * b12
    U[0][11] = 1; U[1][11] = 1
    V[6][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (a10 - a00) * (b02 + b03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (a01 - a11) * (b12 + b13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
