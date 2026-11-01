"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 Strassen multiplications = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for A (2x2)
    # U indices: i*m + j
    # i in {0,1}, j in {0,1}
    # 0: 0,0 ; 1: 0,1 ; 2: 1,0 ; 3: 1,1
    
    # Strassen A factors:
    # S1: A00 + A11
    # S2: A10 + A11
    # S3: A00
    # S4: A11
    # S5: A00 + A01
    # S6: A10 - A00
    # S7: A01 - A11
    
    U_factors = [
        [1, 0, 1, 0], # S1: A00+A11 (indices 0, 3) -> wait, A11 is index 3? 
                      # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3.
                      # S1: A00(0) + A11(3) -> [1, 0, 0, 1]
        [0, 0, 1, 1], # S2: A10(2) + A11(3)
        [1, 0, 0, 0], # S3: A00(0)
        [0, 0, 0, 1], # S4: A11(3)
        [1, 1, 0, 0], # S5: A00(0) + A01(1)
        [-1, 0, 1, 0],# S6: A10(2) - A00(0)
        [0, 1, 0, -1] # S7: A01(1) - A11(3)
    ]
    # Correct U_factors:
    U_factors = [
        [1, 0, 0, 1], # S1
        [0, 0, 1, 1], # S2
        [1, 0, 0, 0], # S3
        [0, 0, 0, 1], # S4
        [1, 1, 0, 0], # S5
        [-1, 0, 1, 0],# S6
        [0, 1, 0, -1] # S7
    ]
    
    # V factors for each block b
    # V indices: j*p + k
    # j in {0,1}, k in {0..15}
    # For block b, k in {2b, 2b+1}
    # B00 = V[2b], B01 = V[2b+1], B10 = V[16+2b], B11 = V[16+2b+1]
    
    # Strassen B factors:
    # T1: B00 + B11
    # T2: B01
    # T3: B10 - B11
    # T4: B00 - B10
    # T5: B01 - B11
    # T6: B00 + B10
    # T7: B01 + B11
    
    # W factors for each block b
    # W indices: k*n + i
    # k in {0..15}, i in {0,1}
    # For block b, k in {2b, 2b+1}
    # C00 = W[2b*2 + 0] = W[4b]
    # C01 = W[(2b+1)*2 + 0] = W[4b+2]
    # C10 = W[2b*2 + 1] = W[4b+1]
    # C11 = W[(2b+1)*2 + 1] = W[4b+3]
    
    # Reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P2 + P5
    # C10 = P3 + P6
    # C11 = P1 - P2 + P3 + P7
    
    # W coefficients for P1..P7 for C00, C01, C10, C11
    W_factors = [
        [1, 0, 0, 1], # P1: C00(1), C01(0), C10(0), C11(1)
        [0, 1, 0, -1],# P2: C00(0), C01(1), C10(0), C11(-1)
        [0, 0, 1, 1], # P3: C00(0), C01(0), C10(1), C11(1)
        [1, 0, 0, 0], # P4: C00(1), C01(0), C10(0), C11(0)
        [-1, 1, 0, 0],# P5: C00(-1), C01(1), C10(0), C11(0)
        [0, 0, 1, 0], # P6: C00(0), C01(0), C10(1), C11(0)
        [1, 0, 0, 1]  # P7: C00(1), C01(0), C10(0), C11(1)
    ]
    
    for b in range(8):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for idx, val in enumerate(U_factors[s]):
                U[idx][col] = val
                
            # Fill V
            # Indices for B in block b
            i_b00 = 0 * p + 2 * b
            i_b01 = 0 * p + 2 * b + 1
            i_b10 = 1 * p + 2 * b
            i_b11 = 1 * p + 2 * b + 1
            
            # T factors
            if s == 0: # T1: B00 + B11
                V[i_b00][col] = 1
                V[i_b11][col] = 1
            elif s == 1: # T2: B01
                V[i_b01][col] = 1
            elif s == 2: # T3: B10 - B11
                V[i_b10][col] = 1
                V[i_b11][col] = -1
            elif s == 3: # T4: B00 - B10
                V[i_b00][col] = 1
                V[i_b10][col] = -1
            elif s == 4: # T5: B01 - B11
                V[i_b01][col] = 1
                V[i_b11][col] = -1
            elif s == 5: # T6: B00 + B10
                V[i_b00][col] = 1
                V[i_b10][col] = 1
            elif s == 6: # T7: B01 + B11
                V[i_b01][col] = 1
                V[i_b11][col] = 1
                
            # Fill W
            # Indices for C in block b
            i_c00 = 2 * b * n + 0
            i_c01 = (2 * b + 1) * n + 0
            i_c10 = 2 * b * n + 1
            i_c11 = (2 * b + 1) * n + 1
            
            w_vals = W_factors[s]
            W[i_c00][col] = w_vals[0]
            W[i_c01][col] = w_vals[1]
            W[i_c10][col] = w_vals[2]
            W[i_c11][col] = w_vals[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
