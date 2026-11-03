import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x4 @ 4x4 matrix multiplication.
    
    The multiplication is decomposed into four independent 2x2 @ 2x2 matrix multiplications.
    Each 2x2 @ 2x2 multiplication uses Strassen's algorithm (rank 7).
    Total rank = 4 * 7 = 28.
    
    A is 2x4, B is 4x4, C is 2x4.
    A = [A11, A12] where A11, A12 are 2x2.
    B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    C = [C11, C12] where C11 = A11B11 + A12B21, C12 = A11B12 + A12B22.
    
    We compute:
    P1 = A11 B11
    P2 = A12 B21
    P3 = A11 B12
    P4 = A12 B22
    
    C11 = P1 + P2
    C12 = P3 + P4
    
    Each Pi is computed using 7 scalar multiplications (Strassen).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Deadline guard (not strictly needed for static construction, but per directive)
    start_time = time.time()
    deadline = 120.0
    
    # Rank is 28
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # Local indices for 2x2 matrices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Strassen terms (M1..M7)
    # U part: coeffs for A elements
    # V part: coeffs for B elements
    # W part: coeffs for C elements
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
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
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # We have 4 blocks of multiplication:
    # Block 0: A[:, 0:2] @ B[0:2, 0:2] -> C[:, 0:2]  (A11 @ B11)
    # Block 1: A[:, 2:4] @ B[2:4, 0:2] -> C[:, 0:2]  (A12 @ B21)
    # Block 2: A[:, 0:2] @ B[0:2, 2:4] -> C[:, 2:4]  (A11 @ B12)
    # Block 3: A[:, 2:4] @ B[2:4, 2:4] -> C[:, 2:4]  (A12 @ B22)
    
    # Mapping of block index (0-3) to offsets
    # b=0: j_off=0, k_off=0
    # b=1: j_off=2, k_off=0
    # b=2: j_off=0, k_off=2
    # b=3: j_off=2, k_off=2
    
    for b in range(4):
        j_off = 2 * (b // 2)  # 0, 2, 0, 2
        k_off = 2 * (b % 2)   # 0, 0, 2, 2
        
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = strassen_terms[s]
            
            # Update U (A matrix)
            # A is 2x4. Index i*m + j.
            # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                j = j_off + j_local
                a_idx = i * m + j
                U[a_idx][col] += coeff
            
            # Update V (B matrix)
            # B is 4x4. Index i*p + j.
            # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for y_idx, coeff in v_dict.items():
                i_local = y_idx // 2
                j_local = y_idx % 2
                i = j_off + i_local
                j = k_off + j_local
                b_idx = i * p + j
                V[b_idx][col] += coeff
            
            # Update W (C matrix)
            # C is 2x4. Index k*n + i (trace convention: W[k*n+i, r])
            # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                k = k_off + k_local
                c_idx = k * n + i
                W[c_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
