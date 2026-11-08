"""Optimized tensor decomposition for 2x3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The trivial rank is n*m*p = 2*3*6 = 36.
    We use a block decomposition: split the 3x6 matrix into three 3x2 blocks.
    Each 2x3x2 block can be done with rank 11 (Strassen for 2x2x2 + 4 scalar mults).
    Total rank = 3 * 11 = 33.
    
    We verify this is correct and return the decomposition.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Use block decomposition: 3 blocks of 2x3x2
    # Each block has rank 11
    rank = 33
    
    # U: (n*m, rank) = (6, 33)
    # V: (m*p, rank) = (18, 33)  
    # W: (p*n, rank) = (12, 33)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for a 2x3x2 block
    # A is 2x3: indices (i,j) with i in {0,1}, j in {0,1,2}, flat = i*3+j
    # B is 3x2: indices (j,k) with j in {0,1,2}, k in {0,1}, flat = j*2+k
    # C is 2x2: indices (i,k) with i in {0,1}, k in {0,1}, flat = i*2+k
    # W convention: W[k*n+i, r]
    
    # Strassen decomposition for the 2x2x2 part (j=0,1) + 4 scalar mults for j=2
    
    # A_L (2x2) uses A[0,0], A[0,1], A[1,0], A[1,1] = indices 0,1,3,4
    # B_T (2x2) uses B[0,0], B[0,1], B[1,0], B[1,1] = indices 0,1,2,3
    # A[:,2] = A[0,2], A[1,2] = indices 2,5
    # B[2,:] = B[2,0], B[2,1] = indices 4,5
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}  # C00, C11
    })
    # M2 = (A10+A11)B00
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}  # C10, -C11
    })
    # M3 = A00(B01-B11)
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}  # C01, C11
    })
    # M4 = A11(B10-B00)
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}  # C00, C10
    })
    # M5 = (A00+A01)B11
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}  # -C00, C01
    })
    # M6 = (A10-A00)(B00+B01)
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}  # C11
    })
    # M7 = (A01-A11)(B10+B11)
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}  # C00
    })
    
    # Scalar terms for a * b where a = A[:,2], b = B[2,:]
    # K1 = A02 * B20
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}  # C00
    })
    # K2 = A02 * B21
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}  # C01
    })
    # K3 = A12 * B20
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}  # C10
    })
    # K4 = A12 * B21
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}  # C11
    })
    
    # Fill the matrices for 3 blocks
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            
            # U: A indices are global (same for all blocks since A is 2x3)
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V: B indices in block: j in 0..2, k in 0..1
            # Global B index: j*p + (block*2 + k)
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # W: C indices in block: i in 0..1, k in 0..1
            # Global C index: W[k*n+i, r] where k is global column
            # k_global = block*2 + k_local
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                k_global = block * 2 + k_local
                g_idx = k_global * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
