import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Rank-28 decomposition using Strassen's 7-term algorithm applied
    # to four 2x2 column-pairs of B.
    #
    # The matrix multiplication tensor M_{2,2,8} decomposes as a direct sum
    # of four M_{2,2,2} tensors (one per column-pair of B), since the B and C
    # indices for different pairs are disjoint. Each M_{2,2,2} has rank 7
    # (Strassen), so total rank = 4 * 7 = 28.
    #
    # Strassen's 7 products for a 2x2 * 2x2 multiplication:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # Output reconstruction:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Tensor decomposition convention:
    #   T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    #   a = i*m + j  (A row i, col j)
    #   b = j*p + k  (B row j, col k)
    #   c = k*n + i  (C row k, col i) [official trace convention]
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen product definitions:
    # Each product is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(i,j): coeff}
    # B_coeffs: dict {(j, k_local): coeff} where k_local in {0,1}
    # C_coeffs: dict {(i, k_local): coeff} where k_local in {0,1}
    
    strassen_products = [
        # M1 = (A00+A11)*(B00+B11)
        # C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)*B00
        # C10 += M2, C11 -= M2
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # M3 = A00*(B01-B11)
        # C01 += M3, C11 += M3
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # M4 = A11*(B10-B00)
        # C00 += M4, C10 += M4
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)*B11
        # C00 -= M5, C01 += M5
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)*(B00+B01)
        # C11 += M6
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7 = (A01-A11)*(B10+B11)
        # C00 += M7
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    for q in range(p // 2):  # 4 column-pairs
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, (a_coeff, b_coeff, c_coeff) in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients, index i*m + j
            for (i, j), coeff in a_coeff.items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients, index j*p + k_global
            for (j, k_local), coeff in b_coeff.items():
                k_global = k1 + k_local
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions, index k_global*n + i
            for (i, k_local), coeff in c_coeff.items():
                k_global = k1 + k_local
                W[k_global * n + i, t] = coeff
    
    # Verify the decomposition is exact
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1
    
    if not np.array_equal(T_computed, T_target):
        # Fallback to schoolbook rank-32
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
