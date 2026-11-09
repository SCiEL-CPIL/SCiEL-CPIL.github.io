import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0

    # Construction: 8 disjoint K4s
    # This graph is C5-free and has independence number 8.
    # n = 32.
    
    n = 32
    adj = [set() for _ in range(n)]
    edges = []
    
    # Iterate over 8 blocks
    for block_idx in range(8):
        start_vertex = block_idx * 4
        # Get the 4 vertices in this block
        block_vertices = [start_vertex + i for i in range(4)]
        
        # Add all edges between these 4 vertices
        for u, v in combinations(block_vertices, 2):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])

    # Verification
    # We know this structure is valid.
    # Quick check:
    # 1. No C5: Disjoint K4s have no cycles of length > 4 within a component, 
    #    and no edges between components. So no C5.
    # 2. No independent set of 9: 
    #    Any independent set can contain at most 1 vertex from each K4.
    #    There are 8 K4s, so max independent set size is 8.
    
    if time.time() - start_time > deadline:
        return {'n': 0, 'edges': []}

    return {'n': n, 'edges': edges}
