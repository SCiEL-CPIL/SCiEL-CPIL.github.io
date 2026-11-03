"""Kronecker-style decomposition of 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    # Split m=5 into 2+3, p=6 into 2+4
    # This gives 4 blocks: (3x2x2), (3x3x2), (3x2x4), (3x3x4)
    
    # For each block, use schoolbook decomposition to ensure correctness
    # Block sizes: (n_i, m_i, p_i)
    blocks = [
        (3, 2, 2),  # i=0..2, j=0..1, k=0..1
        (3, 3, 2),  # i=0..2, j=2..4, k=0..1
        (3, 2, 4),  # i=0..2, j=0..1, k=2..5
        (3, 3, 4),  # i=0..2, j=2..4, k=2..5
    ]
    
    total_rank = sum(b[0] * b[1] * b[2] for b in blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define row indices for each block
    block_offsets = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    col = 0
    for idx, (ni, mi, pi) in enumerate(blocks):
        i_off, j_off, k_off = block_offsets[idx]
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
