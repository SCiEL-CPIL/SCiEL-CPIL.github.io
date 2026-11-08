"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Block decomposition.
    # Dimensions: n=2, m=3, p=6.
    # Split p=6 into 3 blocks of size 2.
    # Each block corresponds to multiplying a 2x3 matrix A_block by a 3x2 matrix B_block to get a 2x2 matrix C_block.
    # We use a rank-11 decomposition for the 2x3x2 tensor, derived from Strassen's 2x2x2 algorithm (rank 7)
    # plus 4 scalar multiplications for the remaining terms involving the 3rd column/row.
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    if p % 2 != 0:
        raise ValueError("This implementation requires p to be even")
        
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    # Initialize matrices
    # U: (n*m, rank) -> (6, total_rank)
    # V: (m*p, rank) -> (18, total_rank)
    # W: (p*n, rank) -> (12, total_rank)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define the 11 terms for a single 2x3x2 block.
    # Local indices:
    # A (2x3): i in 0..1, j in 0..2. Flat index a_idx = i*3 + j.
    # B (3x2): j in 0..2, k in 0..1. Flat index b_idx = j*2 + k.
    # C (2x2): i in 0..1, k in 0..1. Flat index c_idx = i*2 + k.
    
    # Terms are defined as (u_indices, v_indices, w_indices, coeffs)
    # We represent each term by dictionaries mapping local flat index to coefficient.
    
    terms = []
    
    # Strassen-like decomposition for the top-left 2x2 sub-blocks (j=0,1 for A, j=0,1 for B)
    # A_sub = [[A00, A01], [A10, A11]] -> local flat indices: 0, 1, 3, 4
    # B_sub = [[B00, B01], [B10, B11]] -> local flat indices: 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    terms.append({
        'u': {0: 1, 4: 1},
        'v': {0: 1, 3: 1},
        'w': {0: 1, 3: 1}
    })
    
    # M2 = (A10+A11)B00 -> contributes to C10, C11 (with sign flip for C11)
    terms.append({
        'u': {3: 1, 4: 1},
        'v': {0: 1},
        'w': {2: 1, 3: -1}
    })
    
    # M3 = A00(B01-B11) -> contributes to C01, C11
    terms.append({
        'u': {0: 1},
        'v': {1: 1, 3: -1},
        'w': {1: 1, 3: 1}
    })
    
    # M4 = A11(B10-B00) -> contributes to C00, C10
    terms.append({
        'u': {4: 1},
        'v': {2: 1, 0: -1},
        'w': {0: 1, 2: 1}
    })
    
    # M5 = (A00+A01)B11 -> contributes to C00 (with sign flip), C01
    terms.append({
        'u': {0: 1, 1: 1},
        'v': {3: 1},
        'w': {0: -1, 1: 1}
    })
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    terms.append({
        'u': {3: 1, 0: -1},
        'v': {0: 1, 1: 1},
        'w': {3: 1}
    })
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    terms.append({
        'u': {1: 1, 4: -1},
        'v': {2: 1, 3: 1},
        'w': {0: 1}
    })
    
    # Remaining terms for the 3rd column of A (j=2) and 3rd row of B (j=2).
    # A_col2 = [A02, A12] -> local flat indices: 2, 5
    # B_row2 = [B20, B21] -> local flat indices: 4, 5
    
    # K1 = A02 * B20 -> C00
    terms.append({
        'u': {2: 1},
        'v': {4: 1},
        'w': {0: 1}
    })
    
    # K2 = A02 * B21 -> C01
    terms.append({
        'u': {2: 1},
        'v': {5: 1},
        'w': {1: 1}
    })
    
    # K3 = A12 * B20 -> C10
    terms.append({
        'u': {5: 1},
        'v': {4: 1},
        'w': {2: 1}
    })
    
    # K4 = A12 * B21 -> C11
    terms.append({
        'u': {5: 1},
        'v': {5: 1},
        'w': {3: 1}
    })
    
    # Fill the global matrices
    for b in range(num_blocks):
        base_col = b * rank_per_block
        base_k = b * block_size
        
        for t_idx, term in enumerate(terms):
            col = base_col + t_idx
            
            # Fill U
            # U index is global flat index of A: i*m + j
            # Local A flat index: i*3 + j. Since m=3, this is the same.
            for l_idx, val in term['u'].items():
                U[l_idx][col] = val
            
            # Fill V
            # V index is global flat index of B: j*p + k
            # Local B flat index: j*2 + k.
            # j is the middle dimension index (0..2).
            # k is the local column index (0..1).
            # Global k = base_k + k_local.
            for l_idx, val in term['v'].items():
                j_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = j_local * p + (base_k + k_local)
                V[g_idx][col] = val
            
            # Fill W
            # W index is global flat index of C: k*n + i (per trace convention W[k*n+i, r])
            # Local C flat index: i*2 + k.
            # i is the row index (0..1).
            # k is the local column index (0..1).
            # Global k = base_k + k_local.
            for l_idx, val in term['w'].items():
                i_local = l_idx // block_size
                k_local = l_idx % block_size
                g_idx = (base_k + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
