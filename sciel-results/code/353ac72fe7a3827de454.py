"""Rewritten program: exhaustive search for max distinct degrees in induced subgraph."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency sets for fast lookup
N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def _count_distinct_degrees(vertices):
    """Count distinct degrees in the induced subgraph on 'vertices'."""
    vs = set(vertices)
    degrees = []
    for v in vertices:
        deg = sum(1 for u in vs if u != v and u in ADJ[v])
        degrees.append(deg)
    return len(set(degrees))

def _exhaustive_search(deadline):
    """Exhaustively search all subsets of size >= 8, return best vertex list."""
    n = N
    min_size = P['minimum_vertices']
    best_count = 0
    best_vertices = None
    
    # Search from largest subsets down (more vertices can yield more distinct degrees)
    for size in range(n, min_size - 1, -1):
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            cnt = _count_distinct_degrees(combo)
            if cnt > best_count:
                best_count = cnt
                best_vertices = combo
        if time.time() > deadline:
            break
    
    if best_vertices is None:
        # Fallback: try size 8
        for combo in combinations(range(n), min_size):
            cnt = _count_distinct_degrees(combo)
            if cnt > best_count:
                best_count = cnt
                best_vertices = combo
    
    return best_vertices if best_vertices is not None else tuple(range(min_size))

def construct():
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # Start with seed result
    best_vertices = tuple(range(9))
    best_count = _count_distinct_degrees(best_vertices)
    
    # Exhaustive search
    result = _exhaustive_search(deadline)
    if result is not None:
        result_count = _count_distinct_degrees(result)
        if result_count >= best_count:
            best_vertices = result
            best_count = result_count
    
    return list(best_vertices)
# EVOLVE-BLOCK-END
