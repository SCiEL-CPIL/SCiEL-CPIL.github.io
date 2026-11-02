import time

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 blocks * 7 Strassen steps = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Strassen coefficients for V (B matrix)
    # B indices for block b: 
    # B00 (j=0, k=2b), B01 (j=0, k=2b+1)
    # B10 (j=1, k=2b), B11 (j=1, k=2b+1)
    # Global V index: j * p + k
    
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # Strassen coefficients for W (C matrix)
    # C indices for block b:
    # C00 (i=0, k=2b), C01 (i=0, k=2b+1)
    # C10 (i=1, k=2b), C11 (i=1, k=2b+1)
    # Global W index: k * n + i
    
    # C00 = M1 - M4 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M3 - M5 + M6
    
    # W coeffs for each M_s
    # Keys are global W indices? No, local block indices first.
    # We will map local to global in the loop.
    
    W_coeffs_local = [
        # M1: C00(+1), C10(0), C01(0), C11(-1)
        # Wait, C11 = M1 ... so M1 contributes +1 to C11.
        # Let's re-verify C11.
        # C11 = M1 - M3 - M5 + M6. Yes, +1 for M1.
        # So M1: C00: 1, C11: -1? No.
        # C00 = M1 - M4 - M7. M1 coeff is +1.
        # C11 = M1 - M3 - M5 + M6. M1 coeff is +1.
        # So M1 contributes +1 to C00 and +1 to C11.
        
        # Let's list W contributions for each M_s to (C00, C10, C01, C11)
        # M1: (1, 0, 0, 1)
        # M2: (0, 1, 0, 0)
        # M3: (0, 0, 1, -1)
        # M4: (-1, 1, 0, 0)
        # M5: (0, 0, 1, -1)
        # M6: (0, 0, 0, 1)
        # M7: (-1, 0, 0, 0)
        
        [1, 0, 0, 1],   # M1
        [0, 1, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 1, -1],  # M5
        [0, 0, 0, 1],   # M6
        [-1, 0, 0, 0]   # M7
    ]
    
    for b in range(7):
        for s in range(7):
            col = b * 7 + s
            
            # Update U
            for idx, val in U_coeffs[s].items():
                U[idx][col] += val
            
            # Update V
            k0 = 2 * b
            k1 = 2 * b + 1
            # B00: j=0, k=k0 -> 0*14 + k0 = k0
            # B01: j=0, k=k1 -> k1
            # B10: j=1, k=k0 -> 14 + k0
            # B11: j=1, k=k1 -> 14 + k1
            
            if s == 0: # M1
                V[k0][col] += 1
                V[14 + k1][col] += 1
            elif s == 1: # M2
                V[k0][col] += 1
            elif s == 2: # M3
                V[k1][col] += 1
                V[14 + k1][col] -= 1
            elif s == 3: # M4
                V[14 + k0][col] += 1
                V[k0][col] -= 1
            elif s == 4: # M5
                V[14 + k1][col] += 1
            elif s == 5: # M6
                V[k0][col] += 1
                V[k1][col] += 1
            elif s == 6: # M7
                V[14 + k0][col] += 1
                V[14 + k1][col] += 1
            
            # Update W
            # C00: i=0, k=k0 -> k0*2 + 0 = 2*k0
            # C01: i=0, k=k1 -> 2*k1
            # C10: i=1, k=k0 -> 2*k0 + 1
            # C11: i=1, k=k1 -> 2*k1 + 1
            
            w00_idx = 2 * k0
            w01_idx = 2 * k1
            w10_idx = 2 * k0 + 1
            w11_idx = 2 * k1 + 1
            
            coeffs = W_coeffs_local[s]
            W[w00_idx][col] += coeffs[0]
            W[w10_idx][col] += coeffs[1]
            W[w01_idx][col] += coeffs[2]
            W[w11_idx][col] += coeffs[3]
            
    return {"U": U, "V": V, "W": W}
