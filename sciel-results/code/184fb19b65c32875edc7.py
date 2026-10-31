import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the 49-rank optimal decomposition for 4x4x4 matrix multiplication.
    
    The minimum rank for 4x4x4 matrix multiplication is known to be 49.
    This decomposition is constructed by applying Strassen's 7-multiplication 
    algorithm at two levels:
    1. Block level: Treat 4x4 matrices as 2x2 blocks of 2x2 matrices. 
       This requires 7 block multiplications.
    2. Scalar/Entry level: Each 2x2 block multiplication is decomposed 
       using Strassen's algorithm, requiring 7 scalar multiplications per block.
    
    Total rank = 7 * 7 = 49.
    
    The decomposition satisfies the tensor identity:
    T[i, j, k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    Interface:
    U: shape (16, 49)
    V: shape (16, 49)
    W: shape (16, 49)
    Entries are integers.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Define Strassen Coefficients for 2x2 Multiplication ---
    # Strassen's algorithm computes C = A * B using 7 products M0..M6.
    # Indices for 2x2 matrices are flattened: 0->00, 1->01, 2->10, 3->11
    
    # Coefficients for A (input 1)
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # beta[s][i][j] is the coefficient of A[i][j] in the s-th product's A-factor.
    # We represent these as 2x2 lists.
    beta = [
        [[1, 0], [0, 1]],  # s=0: A00=1, A11=1
        [[0, 0], [1, 1]],  # s=1: A10=1, A11=1
        [[1, 0], [0, 0]],  # s=2: A00=1
        [[0, 0], [0, 1]],  # s=3: A11=1
        [[1, 1], [0, 0]],  # s=4: A00=1, A01=1
        [[-1, 0], [1, 0]], # s=5: A10=1, A00=-1
        [[0, 1], [0, -1]]  # s=6: A01=1, A11=-1
    ]
    
    # gamma[s][j][k] is the coefficient of B[j][k] in the s-th product's B-factor.
    gamma = [
        [[1, 0], [0, 1]],  # s=0: B00=1, B11=1
        [[1, 0], [0, 0]],  # s=1: B00=1
        [[0, 1], [0, -1]], # s=2: B01=1, B11=-1
        [[-1, 0], [1, 0]], # s=3: B10=1, B00=-1
        [[0, 0], [0, 1]],  # s=4: B11=1
        [[1, 1], [0, 0]],  # s=5: B00=1, B01=1
        [[0, 0], [1, 1]]   # s=6: B10=1, B11=1
    ]
    
    # delta[s][i][k] is the coefficient of C[i][k] in the s-th product's contribution.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    delta = [
        [[1, 0], [0, 1]],  # s=0: C00=1, C11=1
        [[0, 0], [1, -1]], # s=1: C10=1, C11=-1
        [[0, 1], [0, 1]],  # s=2: C01=1, C11=1
        [[1, 0], [1, 0]],  # s=3: C00=1, C10=1
        [[-1, 1], [0, 0]], # s=4: C00=-1, C01=1
        [[0, 0], [0, 1]],  # s=5: C11=1
        [[1, 0], [0, 0]]   # s=6: C00=1
    ]
    
    # --- Construct the 49-rank decomposition ---
    # The 4x4 multiplication is decomposed into 7 outer Strassen products (s)
    # and 7 inner Strassen products (t) for each block.
    # Total columns = 49.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # 1. Fill U (A matrix coefficients)
            # U[row_u, col] where row_u = i*4 + j
            # i, j are 0..3.
            # Block indices: I = i//2, i' = i%2; J = j//2, j' = j%2
            # Coefficient = beta[s][I][J] * beta[t][i'][j']
            
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
                        
            # 2. Fill V (B matrix coefficients)
            # V[row_v, col] where row_v = j*4 + k
            # Block indices: J = j//2, j' = j%2; K = k//2, k' = k%2
            # Coefficient = gamma[s][J][K] * gamma[t][j'][k']
            
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
                        
            # 3. Fill W (C matrix coefficients)
            # W[row_w, col] where row_w = k*4 + i (per convention W[k*n+i])
            # Block indices: K = k//2, k' = k%2; I = i//2, i' = i%2
            # Coefficient = delta[s][I][K] * delta[t][i'][k']
            
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
