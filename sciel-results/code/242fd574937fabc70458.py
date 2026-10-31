import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the 49-rank recursive Strassen decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
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
    # We define the 7 Strassen products M0..M6.
    # For a 2x2 multiplication C = A * B:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # The linear forms for A (beta), B (gamma), and C (delta) are defined below.
    # Indices: 0->00, 1->01, 2->10, 3->11
    
    # beta[s][i][j] : coefficient of A[i][j] in the s-th Strassen product's A-factor
    # gamma[s][j][k] : coefficient of B[j][k] in the s-th Strassen product's B-factor
    # delta[s][i][k] : coefficient of C[i][k] in the s-th Strassen product's C-contribution
    
    # Using 0-indexed matrix entries:
    # A00=A[0][0], A01=A[0][1], A10=A[1][0], A11=A[1][1]
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Let's encode these as 2x2 matrices of coefficients for the 4 entries (00,01,10,11)
    # Flattened index: 0->00, 1->01, 2->10, 3->11
    
    # beta: A coefficients
    # s=0: A00=1, A11=1
    # s=1: A10=1, A11=1
    # s=2: A00=1
    # s=3: A11=1
    # s=4: A00=1, A01=1
    # s=5: A10=1, A00=-1
    # s=6: A01=1, A11=-1
    beta = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, 1]],  # s=1
        [[1, 0], [0, 0]],  # s=2
        [[0, 0], [0, 1]],  # s=3
        [[1, 1], [0, 0]],  # s=4
        [[-1, 0], [1, 0]], # s=5
        [[0, 1], [0, -1]]  # s=6
    ]
    
    # gamma: B coefficients
    # s=0: B00=1, B11=1
    # s=1: B00=1
    # s=2: B01=1, B11=-1
    # s=3: B10=1, B00=-1
    # s=4: B11=1
    # s=5: B00=1, B01=1
    # s=6: B10=1, B11=1
    gamma = [
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [0, -1]], # s=2
        [[-1, 0], [1, 0]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 0], [1, 1]]   # s=6
    ]
    
    # delta: C coefficients
    # s=0: C00=1, C11=1
    # s=1: C10=1, C11=-1
    # s=2: C01=1, C11=1
    # s=3: C00=1, C10=1
    # s=4: C00=-1, C01=1
    # s=5: C11=1
    # s=6: C00=1
    delta = [
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, -1]], # s=1
        [[0, 1], [0, 1]],  # s=2
        [[1, 0], [1, 0]],  # s=3
        [[-1, 1], [0, 0]], # s=4
        [[0, 0], [0, 1]],  # s=5
        [[1, 0], [0, 0]]   # s=6
    ]
    
    # --- Construct the 49-rank decomposition ---
    # The 4x4 multiplication is decomposed into 7 block-level Strassen products.
    # Each block product is a 2x2 multiplication, which is further decomposed into 7 scalar-level Strassen products.
    # Total 49 terms.
    
    # For each outer Strassen term s (0..6) and inner Strassen term t (0..6):
    # The column index in U,V,W is col = s * 7 + t.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # 1. Fill U (A matrix coefficients)
            # U[row_u, col] where row_u = i*4 + j
            # i, j are 0..3.
            # Block indices: I = i//2, i' = i%2; J = j//2, j' = j%2
            # Coefficient = beta[s][I][J] * (inner Strassen A coeff for t at i', j')
            # Wait, the inner Strassen coefficients are the same for all block positions.
            # The inner Strassen term t has A-coeffs beta[t][i'][j'].
            # So U[i*4+j, col] = beta[s][I][J] * beta[t][i'][j']
            
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
            # Note: delta indices are [row][col] of the 2x2 block.
            # In the inner Strassen, delta[t][i'][k'] gives the coeff for C[i'][k'] in the 2x2 block.
            # So we need to match i' and k' correctly.
            
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
