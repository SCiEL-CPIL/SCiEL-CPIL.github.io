"""Rewritten program using Strassen decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition: 7 multiplications of 3x3 blocks.
    # Rank = 7 * 3 * 3 * 3 = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # I, J, K, L are block indices in {0, 1}
    # s is the step in {0..6}
    
    # beta[s][I][J]: coeff of A[I, J] in L_A[s]
    # gamma[s][K][L]: coeff of B[K, L] in L_B[s]
    # alpha[s][I][K]: coeff of P[s] in C[I, K]
    
    beta = [None] * 7
    gamma = [None] * 7
    alpha = [None] * 7
    
    # Initialize to 0
    for s in range(7):
        beta[s] = [[0]*2 for _ in range(2)]
        gamma[s] = [[0]*2 for _ in range(2)]
        alpha[s] = [[0]*2 for _ in range(2)]
        
    # s=0: (A11+A22)(B11+B22)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][0][1] = 0; alpha[0][1][0] = 0; alpha[0][1][1] = 1 # C11+=M1, C22+=M1
    
    # s=1: (A21+A22)B11
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][0][0] = 0; alpha[1][0][1] = 0; alpha[1][1][0] = 1; alpha[1][1][1] = -1 # C21+=M2, C22-=M2
    
    # s=2: A11(B12-B22)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][0] = 0; alpha[2][0][1] = 1; alpha[2][1][0] = 0; alpha[2][1][1] = 1 # C12+=M3, C22+=M3
    
    # s=3: A22(B21-B11)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][0][1] = 0; alpha[3][1][0] = 1; alpha[3][1][1] = 0 # C11+=M4, C21+=M4
    
    # s=4: (A11+A12)B22
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1; alpha[4][1][0] = 0; alpha[4][1][1] = 0 # C11-=M5, C12+=M5
    
    # s=5: (A21-A11)(B11+B12)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][0][0] = 0; alpha[5][0][1] = 0; alpha[5][1][0] = 0; alpha[5][1][1] = 1 # C22+=M6
    
    # s=6: (A12-A22)(B21+B22)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1; alpha[6][0][1] = 0; alpha[6][1][0] = 0; alpha[6][1][1] = 0 # C11+=M7
    
    # Fill U, V, W
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    r = s * 27 + a * 9 + b * 3 + c
                    
                    # U: A components
                    for I in range(2):
                        for J in range(2):
                            if beta[s][I][J] != 0:
                                i_glob = I * 3 + a
                                j_glob = J * 3 + b
                                U[i_glob * m + j_glob][r] = beta[s][I][J]
                                
                    # V: B components
                    for K in range(2):
                        for L in range(2):
                            if gamma[s][K][L] != 0:
                                j_glob = K * 3 + b
                                k_glob = L * 3 + c
                                V[j_glob * p + k_glob][r] = gamma[s][K][L]
                                
                    # W: C components
                    for I in range(2):
                        for K in range(2):
                            if alpha[s][I][K] != 0:
                                i_glob = I * 3 + a
                                k_glob = K * 3 + c
                                W[k_glob * n + i_glob][r] = alpha[s][I][K]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
