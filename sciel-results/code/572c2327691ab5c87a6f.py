"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 6 (for 4x1x6) + 8 * 12 (for 4x4x6 split into 8 2x2x3 blocks)
    rank_part1 = 6
    rank_part2 = 8 * 12
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: M_{4,1,6} using columns 0..5 of the rank
    # A is 4x1 (col 0), B is 1x6 (row 0)
    # U[i, r] = 1
    # V[j, r] = delta(j, r) where j is 0..5? No, V is 1x6.
    # V is flattened (m*p, rank). m=1, p=6.
    # V[0*6+k, r] = delta(k, r)
    # W[k*4+i, r] = delta(k, r)
    
    for r in range(rank_part1):
        # U: 4x1 matrix. Index i*1 + 0.
        for i in range(n):
            U[i * m + 0][r] = 1
            
        # V: 1x6 matrix. Index 0*6 + k.
        for k in range(p):
            V[0 * p + k][r] = 1 if k == r else 0
            
        # W: 6x4 matrix. Index k*4 + i.
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1 if k == r else 0

    # Part 2: M_{4,4,6} using columns 6..101 of the rank
    # A is 4x4 (cols 1..4), B is 4x6 (rows 1..4)
    # Split into 8 blocks of M_{2,2,3}
    # u: 0..1 (A row block), w: 0..1 (A col block / B row block), v: 0..1 (B col block)
    
    col_offset = rank_part1
    
    for u in range(2):
        for w in range(2):
            for v in range(2):
                # Local M_{2,2,3}
                # i' in 0..1, j' in 0..1, k' in 0..2
                for i_prime in range(2):
                    for j_prime in range(2):
                        for k_prime in range(3):
                            r = col_offset + (u * 4 + w * 2 + v) * 12 + (i_prime * 2 + j_prime) * 3 + k_prime
                            
                            # Global indices
                            i = 2 * u + i_prime
                            j = 2 * w + 1 + j_prime  # A col offset 1
                            k = 3 * v + k_prime
                            
                            # B row index for V
                            j_b = 2 * w + 1 + j_prime # B row offset 1
                            
                            # U[i*m + j, r] = 1
                            U[i * m + j][r] = 1
                            
                            # V[j_b*p + k, r] = 1
                            V[j_b * p + k][r] = 1
                            
                            # W[k*n + i, r] = 1
                            W[k * n + i][r] = 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
