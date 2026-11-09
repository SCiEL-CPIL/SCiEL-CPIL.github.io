"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: 8 disjoint K4 components.
    n = 32.
    alpha = 8 (1 from each K4).
    C5-free: Disjoint union of K4s has no cycles of length 5.
    """
    start_time = time.time()
    deadline = 150.0  # Safety margin
    
    n_components = 8
    size = 4
    n = n_components * size
    
    # Build edges for disjoint K4s
    edge_set = set()
    for c in range(n_components):
        offset = c * size
        for i in range(size):
            for j in range(i + 1, size):
                u = offset + i
                v = offset + j
                # Use tuple to deduplicate if we were adding more, but here they are unique
                edge_set.add((min(u, v), max(u, v)))
        
        if time.time() > deadline - 5:
            break
    
    # Convert to list of lists
    edges = [[u, v] for u, v in sorted(edge_set)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
