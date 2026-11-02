import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 33: 3 blocks * (7 Strassen + 4 Outer)
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U coeffs for i,j in {0,1}x{0,1} -> flat index i*2+j
    # V coeffs for j,k in {0,1}x{0,1} -> flat index j*2+k
    # W coeffs for k,i in {0,1}x{0,1} -> flat index k*2+i
    
    strassen_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [1, 0, -1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [0, 1, -1, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00+C11 (W00=1, W11=1)
        [0, 1, 0, 0], # M2: C10 (W01=1? No, C10 is i=1,k=0 -> W01=1)
        # Wait, W index is k*n+i. n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # C10 is i=1, k=0. So W[1] = 1.
        # Vector: 0 1 0 0. Correct.
        [0, 0, 1, 1], # M3: C01 (i=0,k=1 -> W2=1) + C11 (i=1,k=1 -> W3=1). Vector: 0 0 1 1.
        [-1, 0, 1, 0],# M4: -C00 (W0=-1) + C01 (W2=1). Vector: -1 0 1 0.
        [1, 0, 0, 0], # M5: C00 (W0=1). Vector: 1 0 0 0.
        [0, 1, 0, 1], # M6: C10 (W1=1) + C11 (W3=1). Vector: 0 1 0 1.
        [1, 0, 0, 0]  # M7: C00 (W0=1). Vector: 1 0 0 0.
    ]
    
    # Outer product coefficients for 2x1x2
    # A is 2x1 (i, j=2). B is 1x2 (j=2, k). C is 2x2 (i, k).
    # Terms for each i, k in {0,1}x{0,1}
    # U: A[i, 2]. Only one i is active per term.
    # V: B[2, k]. Only one k is active per term.
    # W: C[i, k]. Only one (i,k) is active per term.
    
    outer_U = [
        [1, 0], # i=0
        [0, 1], # i=1
        [1, 0], # i=0
        [0, 1]  # i=1
    ]
    outer_V = [
        [1, 0], # k=0
        [1, 0], # k=0
        [0, 1], # k=1
        [0, 1]  # k=1
    ]
    outer_W = [
        [1, 0, 0, 0], # i=0, k=0 -> W[0]=1
        [0, 1, 0, 0], # i=1, k=0 -> W[1]=1
        [0, 0, 1, 0], # i=0, k=1 -> W[2]=1
        [0, 0, 0, 1]  # i=1, k=1 -> W[3]=1
    ]

    for t in range(3):
        base_col = t * 11
        # Part 1: Strassen (7 terms)
        for s in range(7):
            col = base_col + s
            # U: i in 0,1; j in 0,1
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = strassen_U[s][i * 2 + j]
            # V: j in 0,1; k in 2t, 2t+1
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * t + kp
                    V[j * p + k_global][col] = strassen_V[s][j * 2 + kp]
            # W: k in 2t, 2t+1; i in 0,1
            for kp in range(2):
                k_global = 2 * t + kp
                for i in range(2):
                    W[k_global * n + i][col] = strassen_W[s][kp * 2 + i]
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

        # Part 2: Outer Product (4 terms)
        for o in range(4):
            col = base_col + 7 + o
            # Determine i, kp for this term
            i_term = o // 2
            kp_term = o % 2
            k_global = 2 * t + kp_term
            
            # U: i in 0,1; j=2
            for i in range(2):
                U[i * m + 2][col] = outer_U[o][i]
            
            # V: j=2; k in 2t, 2t+1
            for kp in range(2):
                k_g = 2 * t + kp
                V[2 * p + k_g][col] = outer_V[o][kp]
            
            # W: k in 2t, 2t+1; i in 0,1
            for kp in range(2):
                k_g = 2 * t + kp
                for i in range(2):
                    W[k_g * n + i][col] = outer_W[o][kp * 2 + i]
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
