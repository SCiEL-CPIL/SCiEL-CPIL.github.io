"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose into blocks based on n and p
    # n=4 -> 2 blocks of size 2
    # p=6 -> 3 blocks of size 2
    # Total 6 blocks
    # Each block is 2x5x2, which is the all-ones tensor, rank 1
    
    num_i_blocks = 2
    num_k_blocks = 3
    rank = num_i_blocks * num_k_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(num_i_blocks):
        for kb in range(num_k_blocks):
            # i range: ib*2 to ib*2+1
            # k range: kb*2 to kb*2+1
            i_start = ib * 2
            i_end = i_start + 2
            k_start = kb * 2
            k_end = k_start + 2
            
            for i in range(i_start, i_end):
                for j in range(m):
                    U[i * m + j][col] = 1
            
            for j in range(m):
                for k in range(k_start, k_end):
                    V[j * p + k][col] = 1
            
            for k in range(k_start, k_end):
                for i in range(i_start, i_end):
                    W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
