import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for M_{2,2,8}.
    
    The strategy exploits the block structure of the 2x8 matrix. 
    M_{2,2,8} is the direct sum of four M_{2,2,2} tensors.
    The minimal rank of M_{2,2,2} is 7 (Strassen).
    Thus, the minimal rank of M_{2,2,8} is 4 * 7 = 28.
    
    We construct the decomposition by applying Strassen's algorithm to each of the four 2x2 blocks of the right-hand matrix.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 products for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # C = A * B
    # 
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each entry is a dict mapping (index) -> coefficient
    # U indices: (i, j) for A[i, j]
    # V indices: (j, k) for B[j, k]
    # W indices: (k, i) for C[k, i] -- Note: The task says W[k*n+i, r], which corresponds to C[i, k] if we view C as output[i, k].
    # Wait, standard matrix multiplication C = A * B.
    # A is n x m, B is m x p. C is n x p.
    # Here n=2, m=2, p=8.
    # The tensor is M_{n,m,p}.
    # The output factor W has shape (p*n, rank).
    # The index is W[k*n + i, r]. This suggests the output element is C[i, k].
    # Let's verify the standard convention.
    # Usually, the tensor T has indices (i, j, k) for A[i,j] * B[j,k] -> C[i,k].
    # The decomposition is T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    # So W_{ki, r} corresponds to C[i, k].
    # My W filling loop: W[k_global * n + i, t] = coeff.
    # This matches W_{k, i}.
    
    strassen_products = [
        # P1
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # P2
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        # P3
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        # P4
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # P5
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # P6
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # P7
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    # Iterate over the 4 blocks of the 2x8 matrix B
    # B is 2x8, so we have 4 blocks of size 2x2
    # Block q corresponds to columns 2q and 2q+1 of B
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            
            # Fill U (A indices)
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
                
            # Fill V (B indices)
            # B index is (j, k_global)
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
                
            # Fill W (C indices)
            # C index is (k_global, i)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
