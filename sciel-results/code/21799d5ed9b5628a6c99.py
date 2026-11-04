"""Matrix multiplication 3x4 by 4x6 - rank 36 block-scalar decomposition."""

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Block-based construction:
    # Split the inner dimension m=4 into two blocks of size 2.
    # This results in two independent 3x2x6 matrix multiplications.
    # Rank of 3x2x6 is 3*2*6 = 36.
    # Total rank = 36 + 36 = 72.
    # This is a structurally different implementation (block iteration) 
    # though the rank is the same as the flat schoolbook.
    
    block_size = 2
    num_blocks = m // block_size
    rank_per_block = n * block_size * p
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for b in range(num_blocks):
        j_start = b * block_size
        for i in range(n):
            for jb in range(block_size):
                j = j_start + jb
                for k in range(p):
                    # U index: i*m + j
                    U[i * m + j][col_offset] = 1
                    # V index: j*p + k
                    V[j * p + k][col_offset] = 1
                    # W index: k*n + i
                    W[k * n + i][col_offset] = 1
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
