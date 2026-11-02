import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The tensor is the direct sum of 7 copies of the 2x2 * 2x2 multiplication tensor.
    The rank of 2x2 * 2x2 is 7 (Strassen).
    The rank of the direct sum is the sum of the ranks: 7 * 7 = 49.
    This is the minimal rank for this tensor.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    num_blocks = p // m # 14 // 2 = 7
    
    # Strassen terms for 2x2 * 2x2
    # A indices: 00, 01, 10, 11 (0, 1, 2, 3)
    # B indices: 00, 01, 10, 11 (0, 1, 2, 3)
    # C indices: 00, 01, 10, 11 (0, 1, 2, 3)
    
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dicts {index: value}
    # Indices map to (row, col) in the 2x2 block:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # M2 = (A10 + A11)B00
        # C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        
        # M5 = (A00 + A01)B11
        # C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += M7
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
            
            # U: shape (n*m, rank)
            # Index in U: i*m + j corresponds to A[i,j]
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff
            
            # V: shape (m*p, rank)
            # Index in V: j*p + k corresponds to B[j,k]
            # Block t covers columns 2*t and 2*t+1
            for b_idx, coeff in b_coeffs.items():
                b_i = b_idx // 2 # Row of B block (0 or 1)
                b_j_local = b_idx % 2 # Col of B block (0 or 1)
                b_k = 2 * t + b_j_local # Global col of B
                V[b_i * p + b_k, col] = coeff
            
            # W: shape (p*n, rank)
            # Index in W: k*n + i corresponds to C[i,k] (Trace convention: W[k*n+i, r])
            # Block t covers columns 2*t and 2*t+1
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2 # Row of C block
                c_j_local = c_idx % 2 # Col of C block
                c_k = 2 * t + c_j_local # Global col of C
                W[c_k * n + c_i, col] = coeff

    # Convert to lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
