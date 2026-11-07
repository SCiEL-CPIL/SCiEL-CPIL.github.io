"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _distinct_degrees(vertex_set, adj):
    """Compute the number of distinct degrees in the induced subgraph."""
    vs = set(vertex_set)
    degrees = set()
    for v in vs:
        deg = sum(1 for u in vs if u in adj[v])
        degrees.add(deg)
    return len(degrees)

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    adj = _build_adj(n, edges)
    
    # Precompute adjacency as bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    # Exhaustive search over all subsets of size >= min_v
    # There are C(17,8) + C(17,9) + ... + C(17,17) subsets
    # C(17,8) = 24310, C(17,9) = 24310, etc. Total is manageable.
    
    best_count = 0
    best_subset = None
    
    # Use bitmask representation
    all_vertices = tuple(range(n))
    
    # We'll iterate over all subsets with size >= min_v
    # To be efficient, we can iterate over all 2^17 = 131072 subsets
    # and filter by popcount >= min_v
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        
        # Count bits
        pc = bin(mask).count('1')
        if pc < min_v:
            continue
        
        # Extract vertices
        vertices = [i for i in range(n) if (mask >> i) & 1]
        
        # Compute distinct degrees
        deg_set = set()
        for v in vertices:
            deg = 0
            for u in vertices:
                if (adj_mask[v] >> u) & 1:
                    deg += 1
            deg_set.add(deg)
        
        count = len(deg_set)
        if count > best_count:
            best_count = count
            best_subset = vertices
    
    # Fallback: if no subset found (shouldn't happen), return a valid subset
    if best_subset is None:
        best_subset = list(range(min_v))
    
    # Ensure types are correct: list of Python ints
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
