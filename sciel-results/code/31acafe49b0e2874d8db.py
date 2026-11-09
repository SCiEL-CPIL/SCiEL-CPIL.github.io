"""
Constructs a C5-free graph on 32 vertices with independence number 8.
This achieves the known upper bound for R(C5, K9) = 33.
"""
# EVOLVE-BLOCK-START
import time

def _build_k4_block(offset, size=4):
    """Generates edges for a K4 component starting at 'offset'."""
    edges = []
    for i in range(size):
        for j in range(i + 1, size):
            edges.append([offset + i, offset + j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Known optimal construction: 8 disjoint copies of K4.
    # n = 8 * 4 = 32.
    # Each K4 has alpha=1, so total alpha = 8.
    # K4 is C5-free.
    
    n = 32
    components = 8
    size_per_component = 4
    
    all_edges = []
    
    # Construct disjoint K4s
    for c in range(components):
        base = c * size_per_component
        comp_edges = _build_k4_block(base, size_per_component)
        all_edges.extend(comp_edges)
        
        # Check time budget
        if time.time() > deadline:
            break
            
    # The graph is valid if:
    # 1. No C5: Disjoint union of C5-free graphs is C5-free. K4 is C5-free.
    # 2. No independent set of size 9: alpha(G) = sum(alpha(G_i)) = 8 * 1 = 8 < 9.
    
    return {'n': n, 'edges': all_edges}
# EVOLVE-BLOCK-END
