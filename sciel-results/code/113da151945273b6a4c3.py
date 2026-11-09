import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    # Part 1: 2x1x10 (Rank 20)
    # Part 2: 2x2x10 (Rank 35, using 5 Strassen blocks of 2x2x2)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A[2x1] * B[1x10]
    # A indices: i in 0..1, j=0. U[i*3+0, c]
    # B indices: j=0, k in 0..9. V[0*10+k, c]
    # C indices: k in 0..9, i in 0..1. W[k*2+i, c]
    col = 0
    for i in range(2):
        for k in range(10):
            U[i * 3 + 0][col] = 1
            V[0 * 10 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[2x2] * B[2x10]
    # A indices: i in 0..1, j in 1..2. U[i*3+j, c]
    # B indices: j in 1..2, k in 0..9. V[j*10+k, c]
    # C indices: k in 0..9, i in 0..1. W[k*2+i, c]
    
    # Strassen coefficients for 2x2
    # u_vecs: list of (index_in_A, value)
    # v_vecs: list of (index_in_B, value)
    # w_vecs: list of (index_in_C, value)
    
    # Local indices for 2x2 block of A: 00, 01, 10, 11
    # Global A indices:
    # 00 -> i=0, j=1 -> 0*3+1 = 1
    # 01 -> i=0, j=2 -> 0*3+2 = 2
    # 10 -> i=1, j=1 -> 1*3+1 = 4
    # 11 -> i=1, j=2 -> 1*3+2 = 5
    
    # Local indices for 2x2 block of B: 00, 01, 10, 11
    # Global B indices for block t (k=2t, 2t+1):
    # 00 -> j=1, k=2t -> 1*10+2t
    # 01 -> j=1, k=2t+1 -> 1*10+2t+1
    # 10 -> j=2, k=2t -> 2*10+2t
    # 11 -> j=2, k=2t+1 -> 2*10+2t+1
    
    # Local indices for 2x2 block of C: 00, 01, 10, 11
    # Global C indices for block t:
    # 00 -> k=2t, i=0 -> 2t*2+0 = 4t
    # 01 -> k=2t+1, i=0 -> (2t+1)*2+0 = 4t+2
    # 10 -> k=2t, i=1 -> 2t*2+1 = 4t+1
    # 11 -> k=2t+1, i=1 -> (2t+1)*2+1 = 4t+3
    
    strassen_u = [
        [(1, 1), (5, 1)],      # M1: A00+A11
        [(4, 1), (5, 1)],      # M2: A10+A11
        [(1, 1)],              # M3: A00
        [(5, 1)],              # M4: A11
        [(1, 1), (2, 1)],      # M5: A00+A01
        [(4, 1), (1, -1)],     # M6: A10-A00
        [(2, 1), (5, -1)]      # M7: A01-A11
    ]
    
    # v and w depend on t, so we generate them in the loop
    
    for t in range(5):
        b_00 = 10 + 2*t
        b_01 = 10 + 2*t + 1
        b_10 = 20 + 2*t
        b_11 = 20 + 2*t + 1
        
        c_00 = 4*t
        c_01 = 4*t + 2
        c_10 = 4*t + 1
        c_11 = 4*t + 3
        
        strassen_v = [
            [(b_00, 1), (b_11, 1)],      # M1: B00+B11
            [(b_00, 1)],                 # M2: B00
            [(b_01, 1), (b_11, -1)],     # M3: B01-B11
            [(b_10, 1), (b_00, -1)],     # M4: B10-B00
            [(b_11, 1)],                 # M5: B11
            [(b_00, 1), (b_01, 1)],      # M6: B00+B01
            [(b_10, 1), (b_11, 1)]       # M7: B10+B11
        ]
        
        strassen_w = [
            [(c_00, 1), (c_11, 1)],      # M1: C00+C11
            [(c_10, 1), (c_11, -1)],     # M2: C10-C11
            [(c_01, 1), (c_11, 1)],      # M3: C01+C11
            [(c_00, 1), (c_10, 1)],      # M4: C00+C10
            [(c_00, -1), (c_01, 1)],     # M5: -C00+C01
            [(c_11, 1)],                 # M6: C11
            [(c_00, 1)]                  # M7: C00
        ]
        
        for s in range(7):
            col = 20 + 5*t + s
            # Fill U
            for idx, val in strassen_u[s]:
                U[idx][col] = val
            # Fill V
            for idx, val in strassen_v[s]:
                V[idx][col] = val
            # Fill W
            for idx, val in strassen_w[s]:
                W[idx][col] = val

    return {"U": U, "V": V, "W": W}
