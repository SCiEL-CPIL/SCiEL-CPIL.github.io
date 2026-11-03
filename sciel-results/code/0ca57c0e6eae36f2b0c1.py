"""6x6 matrix multiplication tensor decomposition using Strassen 2x2 block
partition combined with trivial 3x3 multiplication.
Total rank = 7 * 27 = 189."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block multiplication
    # Each of the 7 multiplications M_s = L_A[s] * L_B[s]
    # L_A[s] = sum_{I,J} beta[s][I][J] * A[I,J]
    # L_B[s] = sum_{K,L} gamma[s][K][L] * B[K,L]
    # C[I,K] += alpha[s][I][K] * M_s
    
    # beta[s][I][J], gamma[s][K][L], alpha[s][I][K] for s in 0..6
    beta = [
        [[1, 0], [0, 1]],  # s=0: A00 + A11
        [[0, 0], [1, 1]],  # s=1: A10 + A11
        [[1, 0], [0, 0]],  # s=2: A00
        [[0, 0], [0, 1]],  # s=3: A11
        [[1, 1], [0, 0]],  # s=4: A00 + A01
        [[-1, 0], [1, 0]], # s=5: -A00 + A10
        [[0, 1], [0, -1]], # s=6: A01 - A11
    ]
    gamma = [
        [[1, 0], [0, 1]],  # s=0: B00 + B11
        [[1, 0], [0, 0]],  # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01 - B11
        [[-1, 0], [1, 0]], # s=3: -B00 + B10
        [[0, 0], [0, 1]],  # s=4: B11
        [[1, 1], [0, 0]],  # s=5: B00 + B01
        [[0, 1], [0, 1]],  # s=6: B10 + B11
    ]
    alpha = [
        [[1, 0], [0, 1]],  # s=0: C00 += M0, C11 += M0
        [[0, 0], [1, -1]], # s=1: C10 += M1, C11 -= M1
        [[0, 1], [0, 1]],  # s=2: C01 += M2, C11 += M2
        [[1, 0], [1, 0]],  # s=3: C00 += M3, C10 += M3
        [[-1, 1], [0, 0]], # s=4: C00 -= M4, C01 += M4
        [[0, 0], [0, 1]],  # s=5: C11 += M5
        [[1, 0], [0, 0]],  # s=6: C00 += M6
    ]
    
    block_size = 3
    
    for s in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    r = s * 27 + i_l * 9 + j_l * 3 + k_l
                    
                    # U: A components
                    for I in range(2):
                        for J in range(2):
                            c = beta[s][I][J]
                            if c != 0:
                                i_glob = I * block_size + i_l
                                j_glob = J * block_size + j_l
                                U[i_glob * m + j_glob][r] = c
                                
                    # V: B components
                    for K in range(2):
                        for L in range(2):
                            c = gamma[s][K][L]
                            if c != 0:
                                j_glob = K * block_size + j_l
                                k_glob = L * block_size + k_l
                                V[j_glob * p + k_glob][r] = c
                                
                    # W: C components
                    # W convention: W[k*n + i, r] where i is row of C, k is col of C
                    for I in range(2):
                        for K in range(2):
                            c = alpha[s][I][K]
                            if c != 0:
                                i_glob = I * block_size + i_l
                                k_glob = K * block_size + k_l
                                W[k_glob * n + i_glob][r] = c
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
