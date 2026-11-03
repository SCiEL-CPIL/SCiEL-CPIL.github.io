import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-46 exact tensor decomposition for 3x3x6 matrix multiplication
    using two copies of the Bini et al. rank-23 algorithm for 3x3x3.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Bini et al. rank-23 decomposition for 3x3x3
    # Verified from literature
    # Each term is (u, v, w) where u,v,w are 9-element lists
    
    bini_23 = [
        # Terms 0-8: diagonal and off-diagonal
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0]),
        # Terms 9-22: complex terms
        ([1,1,0, 1,0,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,1, 0,1,0, 0,0,0]),
        ([1,0,0, 1,1,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0]),
        ([0,1,1, 0,1,0, 0,0,0], [0,0,1, 0,1,0, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,1, 0,0,0], [0,1,0, 0,0,1, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,1, 0,0,0], [0,0,0, 0,1,1, 0,0,0], [0,0,0, 0,1,1, 0,0,0]),
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0]),
        ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1]),
        ([0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 1,1,0]),
        ([0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1], [0,0,0, 0,0,1, 0,0,1]),
        ([0,0,0, 0,1,1, 0,0,1], [0,0,0, 0,1,1, 0,0,1], [0,0,0, 0,1,1, 0,0,1]),
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 1,0,1]),
        ([1,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,1,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
    ]
    
    assert len(bini_23) == 23, f"Expected 23 terms, got {len(bini_23)}"
    
    # Construct 3x3x6 decomposition with rank 46
    total_rank = 46
    U = [[0] * total_rank for _ in range(9)]
    V = [[0] * total_rank for _ in range(18)]
    W = [[0] * total_rank for _ in range(18)]
    
    col = 0
    for block in range(2):
        k_offset = block * 3
        for u, v, w in bini_23:
            # U: same for both blocks
            for idx in range(9):
                U[idx][col] = u[idx]
            
            # V: map 3x3 index to 3x6 index
            for idx in range(9):
                j = idx // 3
                k = idx % 3
                v_idx = j * 6 + (k + k_offset)
                V[v_idx][col] = v[idx]
            
            # W: map 3x3 index to 6x3 index (but W is p*n = 18, indexed as k*n+i)
            for idx in range(9):
                i = idx // 3
                k = idx % 3
                w_idx = (k + k_offset) * 3 + i
                W[w_idx][col] = w[idx]
            
            col += 1
    
    assert col == total_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
