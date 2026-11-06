import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    This is a proven valid configuration for R(C5, K9) > 32.
    
    Design Family: Disjoint union of cliques.
    Constraints:
    - No C5 subgraph: Each component is a clique of size <= 4 (K5 contains C5).
    - No independent set of size 9: The independence number of a disjoint union is the sum of independence numbers of components.
      For K4, alpha=1. With 8 components, alpha=8.
    - Maximize n: Max component size is 4 (due to C5-free constraint on cliques).
      Max components is 8 (due to alpha < 9 constraint).
      Max n = 8 * 4 = 32.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 32
    edges = []
    
    # 8 disjoint blocks of size 4
    for b in range(8):
        if time.time() > deadline:
            break
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    return {'n': n, 'edges': edges}
    # EVOLVE-BLOCK-END
