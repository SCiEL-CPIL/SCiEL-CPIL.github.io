"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency sets for fast degree calculation
_ADJ = [set() for _ in range(P['n'])]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

def _distinct_degrees(vertices):
    """Calculate the number of distinct degrees in the induced subgraph."""
    vset = set(vertices)
    degrees = set()
    for v in vertices:
        deg = len(_ADJ[v] & vset)
        degrees.add(deg)
    return len(degrees)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_count = 0
    best_vertices = list(range(8))
    
    n = P['n']
    all_vertices = list(range(n))
    
    # Precompute all 8-vertex subsets
    all_subsets = list(combinations(all_vertices, 8))
    
    # Score all subsets
    scored = []
    for subset in all_subsets:
        if time.time() > deadline:
            break
        count = _distinct_degrees(subset)
        scored.append((count, subset))
        if count > best_count:
            best_count = count
            best_vertices = list(subset)
    
    if time.time() > deadline:
        return best_vertices
    
    # Try to improve by adding more vertices (up to 17)
    # Start from the best 8-vertex subset and try adding vertices
    for size in range(9, 18):
        if time.time() > deadline:
            break
        
        # Generate all subsets of this size
        # For efficiency, focus on subsets that contain or are near the best so far
        all_subsets_size = list(combinations(all_vertices, size))
        
        for subset in all_subsets_size:
            if time.time() > deadline:
                break
            count = _distinct_degrees(subset)
            if count > best_count:
                best_count = count
                best_vertices = list(subset)
    
    return best_vertices
# EVOLVE-BLOCK-END
