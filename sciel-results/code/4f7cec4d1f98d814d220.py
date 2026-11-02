"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses block decomposition: 3 blocks of 2x3 * 3x3, each with schoolbook rank 9.
    Total rank = 27.
    """
    n, m, p = DIMENSIONS
    # Decompose into 3 blocks of 3 columns each
    block_size = 3
    n_blocks = p // block_size  # 3 blocks
    rank_per_block = n * m * block_size  # 2*3*3 = 9
    rank = n_blocks * rank_per_block  # 27
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(n_blocks):
        for i in range(n):
            for j in range(m):
                for k_local in range(block_size):
                    k = b * block_size + k_local
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
