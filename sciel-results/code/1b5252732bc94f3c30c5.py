"""Matrix multiplication 2,2,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based rank-21 decomposition for 2x2 times 2x6 matrix multiplication.
    
    Applies Strassen's algorithm (7 multiplications for 2x2x2) to each of the
    3 column blocks of the 2x6 matrix, giving rank 3*7 = 21 vs schoolbook 24.
    
    U: shape (n*m, rank) = (4, 21) - coefficients for A entries
    V: shape (m*p, rank) = (12, 21) - coefficients for B entries  
    W: shape (p*n, rank) = (12, 21) - coefficients for C entries (trace convention W[k*n+i, r])
    
    Indexing: A[i,j] stored at U index i*m+j
              B[j,k] stored at V index j*p+k
              C[i,k] stored at W index k*n+i (trace convention)
    """
    n, m, p = DIMENSIONS
    rank = 21  # 3 blocks * 7 Strassen multiplications
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21
    
    # Strassen terms for 2x2 x 2x2 multiplication:
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3: A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4: A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5: (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6: (A10-A00)(B00+B10) -> C10 += M6
    # M7: (A01-A11)(B01+B11) -> C00 += M7, C11 += M7
    #
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    strassen_terms = [
        # (M1): (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # (M2): (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # (M3): A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # (M4): A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # (M5): (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # (M6): (A10-A00)(B00+B10)
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}),
        # (M7): (A01-A11)(B01+B11)
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 3: 1}),
    ]
    
    col = 0
    for block in range(p // 2):  # 3 blocks of 2 columns
        for a_dict, b_dict, c_dict in strassen_terms:
            # Fill U for A entries (same for all blocks)
            for a_idx, coeff in a_dict.items():
                U[a_idx][col] = coeff
            # Fill V for B entries (offset by block*2)
            for b_idx, coeff in b_dict.items():
                V[b_idx + block * 2][col] = coeff
            # Fill W for C entries (offset by block*2, trace convention)
            for c_idx, coeff in c_dict.items():
                # c_idx: 0=C00, 1=C01, 2=C10, 3=C11
                # i = c_idx // 2, k = c_idx % 2 + block*2
                i = c_idx // 2
                k = c_idx % 2 + block * 2
                W[k * n + i][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
