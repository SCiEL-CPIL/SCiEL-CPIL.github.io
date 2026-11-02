"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coeffs for A00, A01, A10, A11
    u_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    # V coeffs for B00, B01, B10, B11
    v_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    # W coeffs for C00, C01, C10, C11
    w_strassen = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 0, 0, 0, 1],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    col = 0
    # Process 3 blocks of 2 columns (0-1, 2-3, 4-5)
    for block in range(3):
        start_k = block * 2
        # A indices: i in 0..1, j in 0..1
        # B indices: j in 0..1, k in start_k..start_k+1
        # C indices: i in 0..1, k in start_k..start_k+1
        
        for i in range(2):
            for j in range(2):
                u_row = i * m + j
                for c in range(7):
                    U[u_row][col + c] = u_strassen[i*2+j][c]
        
        for j in range(2):
            for k in range(2):
                v_row = j * p + (start_k + k)
                for c in range(7):
                    V[v_row][col + c] = v_strassen[j*2+k][c]
        
        for i in range(2):
            for k in range(2):
                w_row = (start_k + k) * n + i
                for c in range(7):
                    W[w_row][col + c] = w_strassen[i*2+k][c]
        
        col += 7

    # Process 1 block of 1 column (6) for the 2x2 part
    start_k = 6
    # Schoolbook for 2x2x1
    # P1=A00B06, P2=A01B16, P3=A10B06, P4=A11B16
    # C06 = P1+P2, C16 = P3+P4
    
    # U
    U[0][col] = 1   # A00
    U[1][col+1] = 1 # A01
    U[2][col+2] = 1 # A10
    U[3][col+3] = 1 # A11
    
    # V
    V[0*p + start_k][col] = 1   # B06
    V[1*p + start_k][col+1] = 1 # B16
    V[0*p + start_k][col+2] = 1 # B06
    V[1*p + start_k][col+3] = 1 # B16
    
    # W
    W[start_k*n + 0][col] = 1   # C06
    W[start_k*n + 0][col+1] = 1 # C06
    W[start_k*n + 1][col+2] = 1 # C16
    W[start_k*n + 1][col+3] = 1 # C16
    
    col += 4

    # Process 2x1x7 part (j=2)
    # Q_ik = A_i2 B_2k
    for i in range(2):
        for k in range(7):
            u_row = i * m + 2
            v_row = 2 * p + k
            w_row = k * n + i
            
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
