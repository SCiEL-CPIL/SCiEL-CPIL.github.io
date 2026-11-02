import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The problem is equivalent to multiplying a 2x2 matrix A by 7 independent 2x2 matrices B_i.
    The minimal rank for 2x2 * 2x2 is 7 (Strassen).
    The total rank is 7 * 7 = 49.
    
    Interface:
    U: shape (n*m, rank) -> (4, 49)
    V: shape (m*p, rank) -> (28, 49)
    W: shape (p*n, rank) -> (28, 49)
    
    Indices:
    A[i, j] corresponds to U[i*m + j, r]
    B[j, k] corresponds to V[j*p + k, r]
    C[i, k] corresponds to W[k*n + i, r] (Trace convention W[k*n+i, r])
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    num_blocks = p // m  # 7 blocks of 2x2
    
    # Strassen's algorithm terms for 2x2 * 2x2
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # coeffs are dicts {index: coefficient}
    # a_index: i*m + j
    # b_index: j*p_block + k_block (local to block) -> mapped to global
    # c_index: k_block*n + i (local to block) -> mapped to global
    
    # Strassen 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    rank = num_blocks * 7
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            for a_idx, coeff in a_coeffs.items():
                # a_idx is 0..3 corresponding to A[i,j] with i=a_idx//2, j=a_idx%2
                # U index is i*m + j = i*2 + j
                U[a_idx, col] = coeff
            
            for b_idx, coeff in b_coeffs.items():
                # b_idx is 0..3 corresponding to B[j,k] with j=b_idx//2, k=b_idx%2
                # V index is j*p + k_global
                # k_global = 2*t + k_local
                b_j = b_idx // 2
                b_k_local = b_idx % 2
                b_k_global = 2 * t + b_k_local
                V[b_j * p + b_k_global, col] = coeff
            
            for c_idx, coeff in c_coeffs.items():
                # c_idx is 0..3 corresponding to C[i,k] with i=c_idx//2, k=c_idx%2?
                # Wait, Strassen output C is 2x2.
                # C00, C01, C10, C11.
                # Indices:
                # 0: C00 -> i=0, k=0
                # 1: C01 -> i=0, k=1
                # 2: C10 -> i=1, k=0
                # 3: C11 -> i=1, k=1
                # W index is k*n + i = k*2 + i
                c_i = c_idx // 2
                c_k_local = c_idx % 2
                c_k_global = 2 * t + c_k_local
                W[c_k_global * n + c_i, col] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
