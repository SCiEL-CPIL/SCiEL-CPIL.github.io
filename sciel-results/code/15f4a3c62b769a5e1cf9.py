import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the 49-rank BCLR decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # beta[s][i][j]: coefficient of A[i][j] in s-th product
    # gamma[s][j][k]: coefficient of B[j][k] in s-th product
    # delta[s][i][k]: coefficient of C[i][k] from s-th product
    
    beta = [
        [[1, 0], [0, 1]],  # M0: (A00+A11)
        [[0, 0], [1, 1]],  # M1: (A10+A11)
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: (A00+A01)
        [[-1, 0], [1, 0]], # M5: (A10-A00)
        [[0, 1], [0, -1]]  # M6: (A01-A11)
    ]
    
    gamma = [
        [[1, 0], [0, 1]],  # M0: (B00+B11)
        [[1, 0], [0, 0]],  # M1: B00
        [[0, 1], [0, -1]], # M2: (B01-B11)
        [[-1, 0], [1, 0]], # M3: (B10-B00)
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: (B00+B01)
        [[0, 0], [1, 1]]   # M6: (B10+B11)
    ]
    
    delta = [
        [[1, 0], [0, 1]],  # M0: C00+=1, C11+=1
        [[0, 0], [1, -1]], # M1: C10+=1, C11-=1
        [[0, 1], [0, 1]],  # M2: C01+=1, C11+=1
        [[1, 0], [1, 0]],  # M3: C00+=1, C10+=1
        [[-1, 1], [0, 0]], # M4: C00-=1, C01+=1
        [[0, 0], [0, 1]],  # M5: C11+=1
        [[1, 0], [0, 0]]   # M6: C00+=1
    ]
    
    # Precompute non-zero entries for speed
    beta_nz = [[(i, j, beta[s][i][j]) for i in range(2) for j in range(2) if beta[s][i][j] != 0] for s in range(7)]
    gamma_nz = [[(j, k, gamma[s][j][k]) for j in range(2) for k in range(2) if gamma[s][j][k] != 0] for s in range(7)]
    delta_nz = [[(i, k, delta[s][i][k]) for i in range(2) for k in range(2) if delta[s][i][k] != 0] for s in range(7)]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Fill U: U[i*4+j, col] = beta[s][I][J] * beta[t][i'][j']
            for I, J, cs in beta_nz[s]:
                for ip, jp, ct in beta_nz[t]:
                    i = I * 2 + ip
                    j = J * 2 + jp
                    U[i * 4 + j][col] = cs * ct
            
            # Fill V: V[j*4+k, col] = gamma[s][J][K] * gamma[t][j'][k']
            for J, K, cs in gamma_nz[s]:
                for jp, kp, ct in gamma_nz[t]:
                    j = J * 2 + jp
                    k = K * 2 + kp
                    V[j * 4 + k][col] = cs * ct
            
            # Fill W: W[k*4+i, col] = delta[s][I][K] * delta[t][i'][k']
            for I, K, cs in delta_nz[s]:
                for ip, kp, ct in delta_nz[t]:
                    i = I * 2 + ip
                    k = K * 2 + kp
                    W[k * 4 + i][col] = cs * ct
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
