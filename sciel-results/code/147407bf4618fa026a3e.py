"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen algorithm for 4x4 matrix multiplication
    # Rank 49 (7x7)
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen vectors for 2x2 multiplication
    # U2[s][i] corresponds to A_{i//2, i%2}
    # V2[s][i] corresponds to B_{i//2, i%2}
    # W2[s][i] corresponds to C_{i//2, i%2}
    
    U2 = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    V2 = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    W2 = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1],  # M3: C01 + C11
        [1, 0, 1, 0],  # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the 49 terms
    for s in range(7):
        for t in range(7):
            idx = s * 7 + t
            
            # Populate U
            for i in range(2):
                for j in range(2):
                    u_coeff = U2[s][2*i + j]
                    if u_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            t_coeff = U2[t][2*u + v]
                            if t_coeff == 0:
                                continue
                            row = 4*(2*i + u) + (2*j + v)
                            U[row][idx] += u_coeff * t_coeff
                            
            # Populate V
            for k in range(2):
                for l in range(2):
                    v_coeff = V2[s][2*k + l]
                    if v_coeff == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            t_coeff = V2[t][2*u + v]
                            if t_coeff == 0:
                                continue
                            row = 4*(2*k + u) + (2*l + v)
                            V[row][idx] += v_coeff * t_coeff
                            
            # Populate W
            for i in range(2):
                for j in range(2):
                    w_coeff_s = W2[s][2*i + j]
                    if w_coeff_s == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            w_coeff_t = W2[t][2*u + v]
                            if w_coeff_t == 0:
                                continue
                            row = 4*(2*i + u) + (2*j + v)
                            W[row][idx] += w_coeff_s * w_coeff_t
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
