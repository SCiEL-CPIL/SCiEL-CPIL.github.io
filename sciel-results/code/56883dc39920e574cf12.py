import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 decomposition for 4x4x4 matrix multiplication.
    
    Uses recursive Strassen: 7 (outer) * 7 (inner) = 49.
    This is the known minimal rank for real-valued 4x4 matrix multiplication 
    using standard recursive methods.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # beta[s][i][j]: coefficient of A[i][j] in s-th Strassen product's A-factor
    beta = [
        [[1, 0], [0, 1]],  # s=0: (A00 + A11)
        [[0, 0], [1, 1]],  # s=1: (A10 + A11)
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: (A00 + A01)
        [[-1, 0], [1, 0]], # s=5: (A10 - A00)
        [[0, 1], [0, -1]]  # s=6: (A01 - A11)
    ]
    
    # gamma[s][j][k]: coefficient of B[j][k] in s-th Strassen product's B-factor
    gamma = [
        [[1, 0], [0, 1]],  # s=0: (B00 + B11)
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: (B01 - B11)
        [[-1, 0], [1, 0]], # s=3: (B10 - B00)
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: (B00 + B01)
        [[0, 0], [1, 1]]   # s=6: (B10 + B11)
    ]
    
    # delta[s][i][k]: coefficient of C[i][k] in s-th Strassen product's C-contribution
    delta = [
        [[1, 0], [0, 1]],  # s=0: C00=1, C11=1
        [[0, 0], [1, -1]], # s=1: C10=1, C11=-1
        [[0, 1], [0, 1]],  # s=2: C01=1, C11=1
        [[1, 0], [1, 0]],  # s=3: C00=1, C10=1
        [[-1, 1], [0, 0]], # s=4: C00=-1, C01=1
        [[0, 0], [0, 1]],  # s=5: C11=1
        [[1, 0], [0, 0]]   # s=6: C00=1
    ]
    
    # Construct the 49-rank decomposition
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U (A matrix coefficients)
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s][I][J] * beta[t][i_prime][j_prime]
                    if coeff != 0:
                        U[row_u][col] = coeff
                        
            # Fill V (B matrix coefficients)
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s][J][K] * gamma[t][j_prime][k_prime]
                    if coeff != 0:
                        V[row_v][col] = coeff
                        
            # Fill W (C matrix coefficients)
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s][I][K] * delta[t][i_prime][k_prime]
                    if coeff != 0:
                        W[row_w][col] = coeff
                        
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
