"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    We use 8 disjoint copies of K4: n=32, alpha=8, trivially C5-free.
    This is the known best lower bound for R(C5, K9).
    """
    # Deadline guard (though construction is instant)
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Check time (not really needed, but follows directive)
    if time.time() > deadline:
        pass # Return best found
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
