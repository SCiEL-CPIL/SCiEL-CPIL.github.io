"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses a block decomposition:
    - n=4 split into [2, 2]
    - m=5 split into [1, 1, 1, 1, 1] (all size-1 middle blocks)
    - p=6 split into [2, 2, 2]
    
    Each 2x1x2 block is handled with schoolbook (rank 4).
    Total blocks: 2 * 5 * 3 = 30, each rank 4, total rank = 120.
    """
    n, m, p = DIMENSIONS
    
    # Block partitions
    i_splits = [2, 2]       # n=4
    j_splits = [1, 1, 1, 1, 1]  # m=5
    k_splits = [2, 2, 2]    # p=6
    
    # Compute offsets
    i_offsets = []
    off = 0
    for s in i_splits:
        i_offsets.append(off)
        off += s
    
    j_offsets = []
    off = 0
    for s in j_splits:
        j_offsets.append(off)
        off += s
    
    k_offsets = []
    off = 0
    for s in k_splits:
        k_offsets.append(off)
        off += s
    
    # Count total rank
    total_rank = 0
    for nb in i_splits:
        for mb in j_splits:
            for pb in k_splits:
                total_rank += nb * mb * pb
    
    # Allocate factor matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    for i_idx in range(len(i_splits)):
        i_0 = i_offsets[i_idx]
        n_b = i_splits[i_idx]
        for j_idx in range(len(j_splits)):
            j_0 = j_offsets[j_idx]
            m_b = j_splits[j_idx]
            for k_idx in range(len(k_splits)):
                k_0 = k_offsets[k_idx]
                p_b = k_splits[k_idx]
                
                # Schoolbook for each block
                for u in range(n_b):
                    for v in range(m_b):
                        for w in range(p_b):
                            U[(i_0 + u) * m + (j_0 + v), col] = 1
                            V[(j_0 + v) * p + (k_0 + w), col] = 1
                            W[(k_0 + w) * n + (i_0 + u), col] = 1
                            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
