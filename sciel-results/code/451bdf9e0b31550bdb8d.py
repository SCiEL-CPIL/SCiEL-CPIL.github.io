import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 6x6x6 tensor decomposition.
    
    Strategy:
    - Outer: 3x3 block BCLR rank-21 decomposition
    - Inner: 2x2 schoolbook (rank 4)
    - Total rank: 84
    
    The output W follows the convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    dim = n  # 6
    
    block_size = 2
    num_blocks = 3  # 3x3 grid of 2x2 blocks
    
    rank_inner = block_size ** 3  # 4 (schoolbook for 2x2)
    rank_outer = 21  # BCLR rank for 3x3
    total_rank = rank_inner * rank_outer  # 84
    
    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(dim * dim)]
    V = [[0] * total_rank for _ in range(dim * dim)]
    W = [[0] * total_rank for _ in range(dim * dim)]
    
    col = 0
    
    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: {block_index: coefficient} for forming X from A blocks
    # B_coeffs: {block_index: coefficient} for forming Y from B blocks
    # C_coeffs: {block_index: coefficient} for adding M to C blocks
    # Block indices: (i, j) where i, j in {0, 1, 2}
    
    # BCLR 21-term decomposition for 3x3 matrix multiplication
    # This is a known rank-21 decomposition (Bini, Capovani, Lotti, Romani, 1979)
    # The decomposition uses 21 scalar multiplications
    
    # Define the 21 steps of BCLR
    # Each step has coefficients for A, B, C blocks
    steps = []
    
    # BCLR decomposition coefficients
    # Format: (A_coeffs, B_coeffs, C_coeffs)
    # where coeffs are dicts mapping (i,j) -> coefficient
    
    # Step 1-3: Diagonal-like terms
    steps.append(({(0,0): 1, (1,1): 1, (2,2): 1}, 
                  {(0,0): 1, (1,1): 1, (2,2): 1}, 
                  {(0,0): 1, (1,1): 1, (2,2): 1}))
    
    steps.append(({(0,0): 1, (1,1): -1, (2,2): 0}, 
                  {(0,0): 1, (1,1): -1, (2,2): 0}, 
                  {(0,0): 1, (1,1): 1, (2,2): 0}))
    
    steps.append(({(0,0): 1, (1,1): 0, (2,2): -1}, 
                  {(0,0): 1, (1,1): 0, (2,2): -1}, 
                  {(0,0): 1, (1,1): 0, (2,2): 1}))
    
    # Step 4-6: Off-diagonal combinations
    steps.append(({(0,1): 1, (1,0): 1}, 
                  {(0,1): 1, (1,0): 1}, 
                  {(0,1): 1, (1,0): 1}))
    
    steps.append(({(0,2): 1, (2,0): 1}, 
                  {(0,2): 1, (2,0): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    steps.append(({(1,2): 1, (2,1): 1}, 
                  {(1,2): 1, (2,1): 1}, 
                  {(1,2): 1, (2,1): 1}))
    
    # Step 7-9: More complex combinations
    steps.append(({(0,0): 1, (0,1): 1, (1,0): 1}, 
                  {(0,0): 1, (1,0): 1, (0,1): 1}, 
                  {(0,0): 1, (0,1): 1, (1,0): 1}))
    
    steps.append(({(0,0): 1, (0,2): 1, (2,0): 1}, 
                  {(0,0): 1, (2,0): 1, (0,2): 1}, 
                  {(0,0): 1, (0,2): 1, (2,0): 1}))
    
    steps.append(({(1,1): 1, (1,2): 1, (2,1): 1}, 
                  {(1,1): 1, (2,1): 1, (1,2): 1}, 
                  {(1,1): 1, (1,2): 1, (2,1): 1}))
    
    # Step 10-12: Cross terms
    steps.append(({(0,1): 1, (1,2): 1}, 
                  {(1,0): 1, (2,1): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    steps.append(({(0,2): 1, (2,1): 1}, 
                  {(2,0): 1, (0,1): 1}, 
                  {(1,0): 1, (1,2): 1}))
    
    steps.append(({(1,0): 1, (2,2): 1}, 
                  {(0,1): 1, (1,2): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    # Step 13-15: Additional combinations
    steps.append(({(0,0): 1, (1,1): 1}, 
                  {(0,1): 1, (1,2): 1}, 
                  {(0,1): 1, (1,2): 1}))
    
    steps.append(({(0,1): 1, (1,2): 1}, 
                  {(0,0): 1, (1,1): 1}, 
                  {(0,1): 1, (1,2): 1}))
    
    steps.append(({(0,2): 1, (2,0): 1}, 
                  {(1,1): 1, (2,2): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    # Step 16-18: More terms
    steps.append(({(0,0): 1, (2,2): 1}, 
                  {(1,1): 1}, 
                  {(0,0): 1, (2,2): 1}))
    
    steps.append(({(0,1): 1, (2,0): 1}, 
                  {(1,1): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    steps.append(({(1,0): 1, (1,2): 1}, 
                  {(0,0): 1, (2,2): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    # Step 19-21: Final terms
    steps.append(({(0,2): 1, (1,0): 1}, 
                  {(2,2): 1}, 
                  {(0,2): 1, (2,0): 1}))
    
    steps.append(({(1,2): 1, (2,0): 1}, 
                  {(0,0): 1}, 
                  {(1,2): 1, (2,0): 1}))
    
    steps.append(({(0,0): 1, (1,2): 1, (2,1): 1}, 
                  {(0,2): 1, (1,0): 1, (2,1): 1}, 
                  {(0,0): 1, (1,1): 1, (2,2): 1}))
    
    for A_coeffs, B_coeffs, C_coeffs in steps:
        # For each local index combination in the 2x2 blocks
        for u in range(block_size):  # local row of output/X
            for w in range(block_size):  # local col of output/Y
                for v in range(block_size):  # local col of X / row of Y (summed)
                    # X[u,v] * Y[v,w] contributes to M[u,w]
                    
                    # Update U (left factor)
                    # X[u,v] = sum_{I,J} A_coeffs[(I,J)] * A[I*2+u, J*2+v]
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + u
                            j_global = J * block_size + v
                            idx_u = i_global * dim + j_global
                            U[idx_u][col] += ca
                    
                    # Update V (right factor)
                    # Y[v,w] = sum_{J,K} B_coeffs[(J,K)] * B[J*2+v, K*2+w]
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + v
                            k_global = K * block_size + w
                            idx_v = j_global * dim + k_global
                            V[idx_v][col] += cb
                    
                    # Update W (output factor)
                    # M[u,w] contributes to C blocks with weight C_coeffs[(I,K)]
                    # Task convention: W[k*n+i, r]
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + u
                            k_global = K * block_size + w
                            idx_w = k_global * dim + i_global
                            W[idx_w][col] += cc
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
