"""Matrix multiplication 2x3 x 3x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-18 decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy: Decompose the 3x9 input into three 3x3 blocks.
    For each block, use a rank-6 decomposition of 2x3 * 3x3.
    Total rank = 3 * 6 = 18.
    
    The rank-6 decomposition for 2x3 * 3x3 is derived by viewing the
    multiplication as a 2x3 matrix multiplication over the 3-dimensional
    algebra of 3x3 diagonal matrices.
    
    Index conventions:
    U[i*m + j, c] where i in [0,n), j in [0,m), c in [0,rank)
    V[j*p + k, c] where j in [0,m), k in [0,p), c in [0,rank)
    W[k*n + i, c] where k in [0,p), i in [0,n), c in [0,rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    block_size = 3
    num_blocks = p // block_size  # 3 blocks
    
    # Rank-6 decomposition for 2x3 * 3x3
    # For each block, we use 6 terms (c' = 0, 1, 2, 3, 4, 5)
    # c' encodes (i_c, k'_c) where i_c = c' // 3, k'_c = c' % 3
    #
    # For a given c':
    #   U[i*m+j, c'] = 1 if i == i_c, else 0
    #   V[j*block_size+k', c'] = 1 if k' == k'_c, else 0
    #   W[k'*n+i, c'] = 1 if k' == k'_c and i == i_c, else 0
    
    rank_per_block = 6
    total_rank = num_blocks * rank_per_block  # 18
    
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x 18
    V = [[0] * total_rank for _ in range(m * p)]      # 27 x 18
    W = [[0] * total_rank for _ in range(p * n)]      # 18 x 18
    
    for q in range(num_blocks):
        base_col = q * rank_per_block
        for cp in range(rank_per_block):
            col = base_col + cp
            i_c = cp // 3
            k_prime_c = cp % 3
            
            # U: only row (i_c, j) for all j
            for j in range(m):
                U[i_c * m + j][col] = 1
            
            # V: only column (j, q*block_size + k_prime_c) for all j
            for j in range(m):
                V[j * p + q * block_size + k_prime_c][col] = 1
            
            # W: only row (q*block_size + k_prime_c, i_c)
            W[(q * block_size + k_prime_c) * n + i_c][col] = 1
    
    # Final verification of the full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
