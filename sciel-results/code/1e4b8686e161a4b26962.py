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
    deadline = time.time() + 115  # safety margin before 120s kill

    n = 32
    block_size = 4
    n_blocks = n // block_size  # 8 blocks

    edges = []
    for block_idx in range(n_blocks):
        offset = block_idx * block_size
        vertices_in_block = range(offset, offset + block_size)
        for u, v in combinations(vertices_in_block, 2):
            edges.append([u, v])

    # Quick structural verification (fast, O(n^2))
    # Check no 5-cycle by confirming each connected component is a clique of size <= 4
    # (a K4 has no cycles longer than 3; a union of K4s has no C5)
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    # Verify: no 5 consecutive vertices in the ordering form a cycle
    # More directly: check that for every 5 distinct vertices, they don't form a C5.
    # Since each component is K4 (4 vertices), any 5 vertices span at least 2 components,
    # and edges only exist within components. So no C5 is possible.
    # Verify alpha <= 8: greedily, one per component max.
    # The construction guarantees this by design.

    if time.time() > deadline:
        # Fallback: return the construction we already built
        pass

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
