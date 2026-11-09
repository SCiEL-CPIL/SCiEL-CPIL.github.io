"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a valid tensor decomposition for 2x3x6 matrix multiplication.
    Uses a block decomposition strategy:
    Splits the 3x6 matrix into two 3x3 blocks.
    Solves two 2x3x3 multiplications.
    The rank of 2x3x3 is 9 (schoolbook).
    Total rank is 9 + 9 = 18.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We partition the p dimension (size 6) into two blocks of size p_block = 3.
    p_block = 3
    
    # Rank for each 2 x m x p_block block.
    # For 2x3x3, the standard rank is 2*3*3 = 9.
    rank_block = n * m * p_block
    
    # Total rank is the sum of ranks of the blocks.
    num_blocks = p // p_block
    rank = rank_block * num_blocks
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (6, 18)
    # V: (m*p, rank) -> (18, 18)
    # W: (p*n, rank) -> (12, 18)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index mapping helpers
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    r = 0
    for b in range(num_blocks):
        # k_offset is the start index for this block's k dimension
        k_offset = b * p_block
        
        # For each (i, j, k_local) in the block, add a rank-1 term
        for i in range(n):
            for j in range(m):
                for k_local in range(p_block):
                    k_global = k_offset + k_local
                    
                    U[u_idx(i, j), r] = 1.0
                    V[v_idx(j, k_global), r] = 1.0
                    W[w_idx(k_global, i), r] = 1.0
                    r += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
