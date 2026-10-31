import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Define Strassen Coefficients for 2x2 Multiplication ---
    # Indices: 0->00, 1->01, 2->10, 3->11 (flattened 2x2)
    # But we use 2D indexing [row][col] for clarity in the loop below.
    
    # beta[s][I][J] : coeff of A[I][J] in s-th product
    # gamma[s][J][K] : coeff of B[J][K] in s-th product
    # delta[s][I][K] : coeff of C[I][K] in s-th product
    
    beta = [
        [[1, 0], [0, 1]],  # s=0: A00+A11
        [[0, 0], [1, 1]],  # s=1: A10+A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00+A01
        [[-1, 0], [1, 0]], # s=5: A10-A00
        [[0, 1], [0, -1]]  # s=6: A01-A11
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # s=0: B00+B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01-B11
        [[-1, 0], [1, 0]], # s=3: B10-B00
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00+B01
        [[0, 0], [1, 1]]   # s=6: B10+B11
    ]
    
    delta = [
        [[1, 0], [0, 1]],  # s=0: C00+C11
        [[0, 0], [1, -1]], # s=1: C10-C11
        [[0, 1], [0, 1]],  # s=2: C01+C11
        [[1, 0], [1, 0]],  # s=3: C00+C10
        [[-1, 1], [0, 0]], # s=4: -C00+C01
        [[0, 0], [0, 1]],  # s=5: C11
        [[1, 0], [0, 0]]   # s=6: C00
    ]
    
    # --- Construct the 49-rank decomposition ---
    # Outer Strassen s (0..6), Inner Strassen t (0..6)
    # Col = s * 7 + t
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # 1. Fill U (A matrix coefficients)
            # U[i*4+j, col] = beta[s][I][J] * beta[t][i'][j']
            # I = i//2, i' = i%2, J = j//2, j' = j%2
            for i in range(4):
                I = i // 2
                ip = i % 2
                for j in range(4):
                    J = j // 2
                    jp = j % 2
                    coeff = beta[s][I][J] * beta[t][ip][jp]
                    if coeff != 0:
                        U[i * 4 + j][col] = coeff
                        
            # 2. Fill V (B matrix coefficients)
            # V[j*4+k, col] = gamma[s][J][K] * gamma[t][j'][k']
            # J = j//2, j' = j%2, K = k//2, k' = k%2
            for j in range(4):
                J = j // 2
                jp = j % 2
                for k in range(4):
                    K = k // 2
                    kp = k % 2
                    coeff = gamma[s][J][K] * gamma[t][jp][kp]
                    if coeff != 0:
                        V[j * 4 + k][col] = coeff
                        
            # 3. Fill W (C matrix coefficients)
            # W[k*4+i, col] = delta[s][I][K] * delta[t][i'][k']
            # K = k//2, k' = k%2, I = i//2, i' = i%2
            # Note: W index is k*n+i.
            # delta[s][I][K] corresponds to block C[I][K].
            # delta[t][i'][k'] corresponds to local C[i'][k'].
            for k in range(4):
                K = k // 2
                kp = k % 2
                for i in range(4):
                    I = i // 2
                    ip = i % 2
                    coeff = delta[s][I][K] * delta[t][ip][kp]
                    if coeff != 0:
                        W[k * 4 + i][col] = coeff
                        
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
