"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    # Block decomposition: view p=9 as 3 blocks of size 3
    block_size = 3
    n_blocks = p // block_size  # 3 blocks
    
    # Trivial per-block: each (2,3,3) block has rank 9
    rank = n * m * block_size * n_blocks  # 2*3*3*3 = 54... that's the same as trivial
    
    # Better: for each block, use rank 9 (trivial for 2x3x3)
    # Total rank = 3 * 9 = 27
    rank = n_blocks * (n * m * block_size)  # 3 * 9 = 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for blk in range(n_blocks):
        for i in range(n):
            for j in range(m):
                for k1 in range(block_size):
                    k = blk * block_size + k1
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
