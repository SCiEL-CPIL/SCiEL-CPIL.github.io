"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Using Strassen for 2x2 blocks of B
    # B is 2x8, split into 4 blocks of 2x2
    # Each block multiplication is 2x2 * 2x2 -> 2x2
    # Strassen rank for 2x2 is 7
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # u[i][j] for A
    u_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    
    # v[j][k'] for B (local k' in 0..1)
    v_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    
    # w[i][k'] for C
    w_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): -1},      # M2
        {(0,1): 1, (1,1): 1},       # M3
        {(0,0): 1, (1,0): 1},       # M4
        {(0,0): -1, (0,1): 1},      # M5
        {(1,1): 1},                 # M6
        {(0,0): 1}                  # M7
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            # Fill U
            for i in range(n):
                for j in range(m):
                    if (i, j) in u_coeffs[s]:
                        U[i * m + j][r] = u_coeffs[s][(i, j)]
            
            # Fill V
            for j in range(m):
                for kp in range(2): # local k'
                    k = q * 2 + kp
                    if (j, kp) in v_coeffs[s]:
                        V[j * p + k][r] = v_coeffs[s][(j, kp)]
            
            # Fill W
            for i in range(n):
                for kp in range(2): # local k'
                    k = q * 2 + kp
                    if (i, kp) in w_coeffs[s]:
                        W[k * n + i][r] = w_coeffs[s][(i, kp)]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
