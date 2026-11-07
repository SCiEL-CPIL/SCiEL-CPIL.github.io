"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 59  # Safety margin within 60s limit
    
    n = P['n']
    half = n // 2
    
    # The optimal graph in the bipartite family is K_{n/2, n/2}.
    # Any graph with more edges must contain a triangle (Mantel's theorem),
    # which typically violates the condition that every edge is critical
    # for maintaining diameter 2 in dense graphs.
    A = list(range(half))
    B = list(range(half, n))
    
    # Construct edges efficiently
    edges = [[u, v] for u in A for v in B]
    
    return edges
# EVOLVE-BLOCK-END
