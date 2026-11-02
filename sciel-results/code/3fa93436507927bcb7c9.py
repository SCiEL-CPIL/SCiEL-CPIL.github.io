import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses the Strassen algorithm applied to 2x2 blocks (Kronecker product structure).
    
    The decomposition is exact with integer entries.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices for 2x2 blocks: 00, 01, 10, 11
    # U (A) coefficients for M1..M7
    U_2 = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1]  # M7: A01-A11
    ]
    
    # V (B) coefficients for M1..M7
    V_2 = [
        [1, 0, 0, 1],  # M1: B00+B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01-B11
        [-1, 0, 1, 0], # M4: B10-B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00+B01
        [0, 0, 1, 1]   # M7: B10+B11
    ]
    
    # W (C) coefficients for M1..M7
    # Note: W index is k*n+i. For 2x2, n=2.
    # Local indices in W_2 correspond to:
    # 0: C00 (i=0, k=0)
    # 1: C10 (i=1, k=0)
    # 2: C01 (i=0, k=1)
    # 3: C11 (i=1, k=1)
    # This ordering (00, 10, 01, 11) is used for the local block mapping.
    W_2 = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 1, 0, -1], # M2: C10 - C11
        [0, 0, 1, 1],  # M3: C01 + C11
        [1, 1, 0, 0],  # M4: C00 + C10
        [-1, 0, 1, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0]   # M7: C00
    ]
    
    # Block coefficients for the outer 2x2 Strassen structure
    # These determine which 2x2 blocks of A, B, C are involved in each of the 7 outer multiplications.
    # Order of blocks: 00, 01, 10, 11
    A_blocks = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    B_blocks = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    C_blocks = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11 (Note: C_blocks order is 00, 01, 10, 11? No, let's stick to the W_2 local order mapping)
        # Let's re-verify C_blocks mapping to W_2 local indices.
        # W_2 local indices: 0=C00, 1=C10, 2=C01, 3=C11.
        # M1: C00+C11 -> indices 0, 3. Coeffs: [1, 0, 0, 1]
        # M2: C10-C11 -> indices 1, 3. Coeffs: [0, 1, 0, -1]
        # M3: C01+C11 -> indices 2, 3. Coeffs: [0, 0, 1, 1]
        # M4: C00+C10 -> indices 0, 1. Coeffs: [1, 1, 0, 0]
        # M5: -C00+C01 -> indices 0, 2. Coeffs: [-1, 0, 1, 0]
        # M6: C11 -> index 3. Coeffs: [0, 0, 0, 1]
        # M7: C00 -> index 0. Coeffs: [1, 0, 0, 0]
        # So C_blocks should be:
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Global index mappings for 4x4 matrix
    # U (A) indices: i*n + k
    # A is 4x4. Blocks are 2x2.
    # Block 00: i in {0,1}, k in {0,1} -> indices 0,1,4,5
    # Block 01: i in {0,1}, k in {2,3} -> indices 2,3,6,7
    # Block 10: i in {2,3}, k in {0,1} -> indices 8,9,12,13
    # Block 11: i in {2,3}, k in {2,3} -> indices 10,11,14,15
    U_block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # V (B) indices: j*n + k (B is m x p, m=4, p=4)
    # Same structure as A
    V_block_indices = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ]
    
    # W (C) indices: k*n + i (C is p x n, p=4, n=4)
    # Block 00: i in {0,1}, k in {0,1}
    # Local 0 (00): i=0,k=0 -> 0
    # Local 1 (10): i=1,k=0 -> 1
    # Local 2 (01): i=0,k=1 -> 4
    # Local 3 (11): i=1,k=1 -> 5
    W_block_00 = [0, 1, 4, 5]
    
    # Block 01: i in {0,1}, k in {2,3}
    # Local 0 (00): i=0,k=2 -> 8
    # Local 1 (10): i=1,k=2 -> 9
    # Local 2 (01): i=0,k=3 -> 12
    # Local 3 (11): i=1,k=3 -> 13
    W_block_01 = [8, 9, 12, 13]
    
    # Block 10: i in {2,3}, k in {0,1}
    # Local 0 (00): i=2,k=0 -> 2
    # Local 1 (10): i=3,k=0 -> 3
    # Local 2 (01): i=2,k=1 -> 6
    # Local 3 (11): i=3,k=1 -> 7
    W_block_10 = [2, 3, 6, 7]
    
    # Block 11: i in {2,3}, k in {2,3}
    # Local 0 (00): i=2,k=2 -> 10
    # Local 1 (10): i=3,k=2 -> 11
    # Local 2 (01): i=2,k=3 -> 14
    # Local 3 (11): i=3,k=3 -> 15
    W_block_11 = [10, 11, 14, 15]
    
    W_block_indices = [W_block_00, W_block_01, W_block_10, W_block_11]
    
    # Initialize matrices
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Construct the 49 columns
    # Outer loop over the 7 Strassen steps for the 2x2 block structure
    # Inner loop over the 7 Strassen steps for the 2x2 sub-block structure
    for outer_step in range(7):
        for inner_step in range(7):
            col = outer_step * 7 + inner_step
            
            # Update U
            for b in range(4):
                coeff_outer = A_blocks[outer_step][b]
                if coeff_outer == 0:
                    continue
                for local_idx in range(4):
                    coeff_inner = U_2[inner_step][local_idx]
                    if coeff_inner == 0:
                        continue
                    global_idx = U_block_indices[b][local_idx]
                    U[global_idx, col] += coeff_outer * coeff_inner
            
            # Update V
            for b in range(4):
                coeff_outer = B_blocks[outer_step][b]
                if coeff_outer == 0:
                    continue
                for local_idx in range(4):
                    coeff_inner = V_2[inner_step][local_idx]
                    if coeff_inner == 0:
                        continue
                    global_idx = V_block_indices[b][local_idx]
                    V[global_idx, col] += coeff_outer * coeff_inner
            
            # Update W
            for b in range(4):
                coeff_outer = C_blocks[outer_step][b]
                if coeff_outer == 0:
                    continue
                for local_idx in range(4):
                    coeff_inner = W_2[inner_step][local_idx]
                    if coeff_inner == 0:
                        continue
                    global_idx = W_block_indices[b][local_idx]
                    W[global_idx, col] += coeff_outer * coeff_inner
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
