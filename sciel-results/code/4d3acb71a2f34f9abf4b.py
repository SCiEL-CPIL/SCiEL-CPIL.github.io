"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-144 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Structure:
    - Two 4x4x4 blocks (B columns 0-3 and 4-7) using Schoolbook (rank 64 each)
    - One 4x4x1 block (B column 8) using Schoolbook (rank 16)
    Total rank: 64 + 64 + 16 = 144
    
    This is a correct, albeit suboptimal, decomposition that fixes the previous
    Strassen implementation errors.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Calculate total rank: n*m*p
    # We can decompose the full 4x4*4x9 multiplication into schoolbook terms
    # Rank = n * m * p = 4 * 4 * 9 = 144
    
    total_rank = n * m * p
    U = [[0] * total_rank for _ in range(n * m)]   # 16 x 144
    V = [[0] * total_rank for _ in range(m * p)]   # 36 x 144
    W = [[0] * total_rank for _ in range(p * n)]   # 36 x 144

    # Schoolbook decomposition:
    # For each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # C[i, k] += A[i, j] * B[j, k]
    # In flat indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Rank index: (i*m + j)*p + k  (or any unique mapping)
    
    rank_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][rank_idx] = 1
                V[v_idx][rank_idx] = 1
                W[w_idx][rank_idx] = 1
                rank_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
