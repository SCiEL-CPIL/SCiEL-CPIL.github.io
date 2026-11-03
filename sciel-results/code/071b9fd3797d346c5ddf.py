"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # We use 12 blocks of 2x2x2 Strassen
    # 12 * 7 = 84 rank
    rank = 12 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Indices 0..3 for U (A), 0..3 for V (B), 0..3 for W (C)
    # U: A00, A01, A10, A11
    # V: B00, B01, B10, B11
    # W: C00, C01, C10, C11 (mapped to W[k*n+i] -> C[i,k] in local? No, W index is k*n+i.
    # In local 2x2, n=2. W index: k*2+i.
    # C00 -> k=0, i=0 -> idx 0
    # C01 -> k=1, i=0 -> idx 2
    # C10 -> k=0, i=1 -> idx 1
    # C11 -> k=1, i=1 -> idx 3
    
    # Strassen terms P1..P7
    # U coeffs (A00, A01, A10, A11)
    U_strassen = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00+A01
        [-1, 0, 1, 0],# P6: A10-A00
        [0, 1, 0, -1] # P7: A01-A11
    ]
    
    # V coeffs (B00, B01, B10, B11)
    V_strassen = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1],# P3: B01-B11
        [-1, 0, 1, 0],# P4: B10-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    # W coeffs (C00, C10, C01, C11) -> Indices 0, 1, 2, 3
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
    
    W_strassen = [
        [1, 0, 0, 1], # P1: C00=1, C11=1
        [0, 1, 0, -1],# P2: C10=1, C11=-1
        [0, 0, 1, 1], # P3: C01=1, C11=1
        [1, 1, 0, 0], # P4: C00=1, C10=1
        [-1, 0, 1, 0],# P5: C00=-1, C01=1
        [0, 0, 0, 1], # P6: C11=1
        [1, 0, 0, 0]  # P7: C00=1
    ]
    
    col = 0
    for i0 in range(2):
        for k0 in range(2):
            for j0 in range(3):
                for t in range(7):
                    # Update U
                    # Global A index: I = 2*i0 + i, K = 2*k0 + k
                    # U row: I * m + K
                    for i in range(2):
                        for k in range(2):
                            I = 2 * i0 + i
                            K = 2 * k0 + k
                            u_row = I * m + K
                            coeff = U_strassen[t][i * 2 + k]
                            if coeff != 0:
                                U[u_row][col] += coeff
                                
                    # Update V
                    # Global B index: K = 2*k0 + k, J = 2*j0 + j
                    # V row: K * p + J
                    for k in range(2):
                        for j in range(2):
                            K = 2 * k0 + k
                            J = 2 * j0 + j
                            v_row = K * p + J
                            coeff = V_strassen[t][k * 2 + j]
                            if coeff != 0:
                                V[v_row][col] += coeff
                                
                    # Update W
                    # Global C index: J = 2*j0 + j, I = 2*i0 + i
                    # W row: J * n + I
                    for j in range(2):
                        for i in range(2):
                            J = 2 * j0 + j
                            I = 2 * i0 + i
                            w_row = J * n + I
                            coeff = W_strassen[t][j * 2 + i]
                            if coeff != 0:
                                W[w_row][col] += coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
