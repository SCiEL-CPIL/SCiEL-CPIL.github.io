import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C5-free graph with no independent set of 9 vertices.
    Uses 8 disjoint cliques of size 4 (8 x K4), giving n = 32.
    
    Verification:
    - Each K4 component has girth 3 (triangle), so no C5 exists.
    - Maximum independent set size = 8 (at most one vertex per K4 component).
    - Since 8 < 9, no independent 9-set exists.
    """
    n = 32
    block_size = 4
    n_blocks = n // block_size  # 8 blocks

    edges = []
    for block_idx in range(n_blocks):
        offset = block_idx * block_size
        vertices_in_block = range(offset, offset + block_size)
        for u, v in combinations(vertices_in_block, 2):
            edges.append([u, v])

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
