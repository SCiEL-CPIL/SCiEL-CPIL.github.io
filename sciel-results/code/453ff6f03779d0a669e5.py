"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 18, 'edge_order': [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [6, 14, 17], [11, 12, 16], [3, 10, 17], [10, 12, 14], [3, 9, 13], [0, 7, 13], [3, 14, 16], [1, 5, 8], [0, 9, 12], [9, 13, 17], [4, 11, 17], [4, 8, 13], [3, 14, 17], [3, 11, 15], [11, 13, 17], [8, 11, 12], [4, 9, 13], [2, 7, 9], [3, 6, 14], [0, 3, 5], [5, 7, 16], [2, 9, 16], [6, 7, 9], [1, 7, 8], [3, 10, 13], [3, 10, 14], [1, 10, 15], [2, 4, 14], [4, 11, 14], [6, 7, 10], [5, 6, 12], [3, 13, 17], [0, 5, 7], [5, 7, 11], [0, 2, 11], [9, 15, 16], [0, 2, 14], [5, 15, 17], [0, 3, 7], [10, 12, 15], [7, 9, 11], [4, 7, 17], [4, 11, 16], [3, 6, 17], [7, 11, 14], [1, 3, 8], [1, 10, 16], [3, 9, 16], [7, 8, 11], [7, 10, 14], [4, 10, 14], [3, 14, 15], [4, 14, 17], [4, 16, 17], [9, 12, 15], [0, 2, 4], [7, 9, 15], [10, 12, 17], [7, 8, 13], [8, 11, 15], [8, 16, 17], [5, 11, 13], [1, 11, 14], [0, 6, 10], [0, 8, 14], [5, 6, 16], [0, 6, 17], [1, 16, 17], [7, 11, 16], [2, 11, 16], [0, 1, 13], [6, 7, 14], [4, 7, 11], [6, 11, 14], [1, 7, 14], [1, 4, 8], [5, 9, 15], [8, 9, 15], [3, 11, 13], [2, 11, 14], [3, 12, 13], [2, 4, 8], [12, 14, 15], [1, 12, 17], [1, 14, 17], [6, 15, 16], [0, 11, 13], [2, 5, 15], [11, 12, 17], [5, 15, 16], [2, 6, 15], [5, 6, 15], [5, 8, 15], [0, 7, 9], [1, 4, 14], [9, 14, 15], [2, 4, 10], [3, 7, 15], [5, 7, 17], [1, 10, 14], [6, 8, 17], [1, 2, 15], [1, 14, 15], [2, 5, 17], [7, 9, 16], [10, 13, 17], [5, 9, 14], [0, 12, 13], [0, 13, 14], [7, 10, 12], [2, 11, 15], [4, 5, 10], [10, 16, 17], [4, 14, 16], [1, 8, 15], [5, 6, 14], [7, 11, 15], [3, 11, 12], [7, 9, 17], [4, 7, 14], [1, 5, 15], [3, 7, 12], [4, 5, 12], [3, 4, 15], [1, 6, 10], [1, 8, 10], [6, 10, 13], [9, 11, 17], [10, 14, 15], [2, 3, 5], [0, 5, 6], [4, 11, 12], [2, 6, 16], [2, 8, 10], [2, 3, 11], [9, 10, 11], [2, 9, 17], [0, 12, 16], [0, 6, 13], [1, 9, 13], [5, 6, 9], [4, 5, 17], [2, 7, 10], [2, 3, 7], [1, 7, 11], [3, 7, 17], [0, 2, 7], [3, 5, 10], [4, 7, 15], [1, 2, 11], [6, 8, 10], [2, 4, 6], [0, 2, 13], [2, 3, 17], [3, 7, 16], [3, 4, 10], [1, 6, 7], [6, 10, 12], [4, 6, 7], [7, 10, 17], [8, 13, 17], [1, 12, 15], [0, 8, 15], [2, 5, 7], [1, 3, 15], [2, 4, 11], [4, 6, 8], [9, 14, 16], [0, 13, 16], [0, 9, 16], [8, 10, 14], [3, 5, 12], [2, 9, 13], [2, 7, 12], [0, 1, 11], [1, 7, 13], [1, 4, 7], [0, 9, 17], [0, 1, 14], [8, 13, 14], [0, 2, 5], [1, 10, 12], [8, 12, 13], [1, 14, 16], [3, 8, 12], [5, 7, 14], [1, 5, 9], [5, 13, 16], [0, 1, 15], [0, 16, 17], [3, 5, 6], [5, 8, 11], [0, 3, 13], [0, 12, 14], [5, 6, 8], [5, 7, 12], [4, 5, 8], [0, 10, 11], [1, 8, 11], [1, 13, 16], [3, 12, 16], [3, 6, 8], [0, 2, 17], [9, 10, 12], [2, 5, 8], [1, 12, 14], [8, 9, 13], [0, 7, 8], [0, 14, 17], [4, 10, 13], [9, 12, 13], [5, 11, 15], [13, 14, 15], [1, 9, 10], [2, 7, 15], [6, 9, 10], [0, 12, 15], [12, 13, 16], [0, 9, 14], [6, 13, 15], [1, 4, 10], [4, 7, 16], [12, 15, 17], [2, 7, 14], [0, 4, 5], [3, 16, 17], [13, 15, 16], [5, 7, 10], [6, 7, 16], [6, 15, 17], [6, 12, 13], [1, 13, 15], [0, 4, 8], [7, 13, 15], [4, 9, 11], [6, 7, 13], [0, 8, 16], [5, 6, 11], [2, 4, 17], [3, 13, 14], [10, 11, 13], [2, 11, 12], [6, 7, 17], [4, 13, 15], [3, 5, 13], [0, 9, 15], [3, 13, 16], [4, 12, 15], [12, 16, 17], [2, 6, 12], [7, 8, 12], [0, 7, 17], [2, 10, 12], [8, 9, 16], [8, 10, 13], [6, 9, 16], [6, 13, 17], [1, 3, 13], [6, 11, 12], [6, 13, 14], [2, 4, 9], [0, 10, 15], [0, 14, 16], [1, 9, 17], [4, 8, 10], [3, 6, 13], [4, 6, 17], [1, 2, 8], [2, 10, 17], [2, 8, 16], [0, 7, 16], [0, 10, 17], [2, 14, 15], [6, 7, 11], [10, 11, 16], [6, 9, 11], [1, 8, 17], [1, 7, 9], [0, 6, 9], [10, 14, 17], [2, 5, 10], [4, 8, 9], [7, 8, 15], [2, 12, 17], [9, 10, 14], [8, 12, 17], [6, 7, 12], [5, 8, 14], [6, 9, 14], [5, 12, 14], [3, 15, 17], [10, 13, 16], [8, 14, 15], [7, 11, 17], [5, 10, 13], [5, 11, 14], [6, 12, 15], [2, 10, 15], [0, 5, 15], [12, 14, 17], [11, 13, 16], [2, 13, 16], [2, 9, 15], [0, 15, 17], [0, 6, 11], [2, 8, 13], [0, 1, 4], [4, 12, 13], [5, 9, 12], [0, 9, 10], [1, 9, 11], [7, 15, 16], [0, 1, 8], [5, 7, 9], [0, 1, 9], [11, 12, 14], [1, 8, 14], [6, 8, 12], [5, 8, 17], [3, 8, 15], [3, 5, 17], [2, 5, 11], [0, 3, 12], [5, 14, 16], [7, 14, 17], [6, 9, 12], [9, 10, 15], [7, 9, 13], [9, 11, 16], [3, 7, 11], [3, 9, 12], [3, 4, 7], [1, 5, 17], [0, 1, 3], [5, 10, 12], [2, 3, 8], [7, 13, 14], [0, 7, 12], [9, 13, 15], [3, 11, 14], [0, 4, 16], [12, 13, 15], [7, 9, 14], [2, 10, 16], [7, 9, 10], [3, 10, 12], [14, 15, 16], [5, 12, 16], [1, 3, 10], [4, 10, 15], [5, 6, 13], [1, 7, 10], [6, 9, 13], [2, 9, 11], [1, 7, 15], [1, 7, 16], [3, 10, 16], [1, 6, 13], [0, 9, 13], [5, 8, 13], [11, 13, 15], [2, 4, 15], [10, 11, 14], [13, 16, 17], [3, 5, 7], [0, 8, 13], [3, 8, 17], [2, 5, 6], [3, 4, 6], [0, 1, 10], [3, 8, 10], [0, 3, 8], [2, 13, 17], [0, 4, 15], [2, 7, 13], [3, 12, 17], [8, 10, 11], [4, 8, 12], [10, 12, 13], [11, 12, 15], [0, 4, 10], [8, 10, 17], [4, 5, 11], [0, 10, 16], [0, 6, 7], [3, 10, 11], [1, 15, 17], [3, 8, 13], [3, 6, 9], [4, 15, 17], [2, 4, 13], [4, 8, 11], [1, 10, 17], [0, 2, 12], [0, 7, 10], [0, 1, 16], [0, 7, 14], [4, 5, 6], [4, 5, 9], [2, 6, 10], [2, 6, 17], [0, 7, 15], [4, 5, 14], [3, 7, 10], [2, 16, 17], [5, 8, 12], [6, 11, 16], [6, 10, 15], [2, 15, 17], [4, 6, 14], [0, 10, 12], [10, 15, 17], [1, 6, 16], [0, 12, 17], [6, 11, 13], [1, 3, 9], [7, 12, 16], [3, 5, 14], [6, 10, 11], [3, 11, 16], [1, 4, 12], [7, 15, 17], [4, 9, 17], [7, 8, 17], [1, 8, 13], [3, 6, 12], [1, 13, 17], [2, 9, 10], [7, 9, 12], [1, 8, 9], [2, 8, 17], [0, 11, 14], [3, 6, 11], [2, 5, 12], [11, 15, 16], [1, 9, 12], [2, 7, 16], [5, 8, 10], [2, 6, 14], [9, 13, 14], [5, 9, 10], [2, 12, 14], [0, 8, 11], [8, 12, 16], [3, 4, 8], [7, 11, 13], [5, 14, 15], [2, 4, 16], [1, 2, 10], [11, 14, 15], [6, 8, 15], [10, 11, 15], [6, 8, 16], [11, 12, 13], [8, 11, 17], [11, 15, 17], [2, 3, 13], [1, 5, 11], [0, 5, 16], [4, 6, 15], [14, 16, 17], [0, 9, 11], [0, 3, 10], [0, 2, 3], [3, 10, 15], [10, 12, 16], [2, 11, 13], [4, 5, 7], [10, 15, 16], [8, 14, 17], [1, 13, 14], [2, 6, 8], [1, 2, 17], [2, 4, 5], [9, 13, 16], [4, 10, 12], [7, 14, 16], [5, 7, 13], [3, 4, 16], [2, 12, 15], [0, 3, 4], [6, 7, 15], [8, 11, 14], [5, 9, 13], [1, 2, 9], [0, 8, 12], [1, 5, 16], [0, 5, 14], [8, 10, 16], [1, 10, 13], [10, 11, 12], [4, 15, 16], [5, 13, 15], [1, 10, 11], [13, 14, 16], [2, 10, 14], [1, 5, 7], [6, 14, 16], [4, 6, 11], [2, 3, 12], [7, 14, 15], [7, 10, 16], [4, 12, 17], [0, 8, 10], [0, 7, 11], [1, 9, 14], [8, 13, 16], [4, 13, 16], [1, 8, 12], [1, 6, 14], [9, 11, 14], [8, 10, 12], [1, 3, 5], [0, 3, 14], [0, 4, 7], [2, 13, 14], [5, 11, 17], [1, 2, 16], [3, 7, 8], [2, 11, 17], [5, 13, 14], [3, 5, 11], [4, 8, 14], [1, 3, 17], [0, 14, 15], [5, 8, 9], [0, 15, 16], [0, 11, 12], [0, 2, 16], [2, 3, 9], [2, 5, 13], [8, 12, 15], [9, 10, 13], [7, 10, 13], [0, 6, 12], [3, 11, 17], [1, 5, 13], [3, 7, 9], [0, 1, 2], [1, 12, 13], [4, 9, 10], [1, 4, 6], [8, 9, 11], [3, 5, 15], [2, 14, 17], [2, 9, 14], [2, 8, 11], [5, 12, 13], [7, 10, 11], [9, 12, 16], [5, 12, 17], [0, 5, 10], [10, 14, 16], [5, 10, 15], [8, 9, 17], [1, 6, 15], [1, 8, 16], [6, 13, 16], [4, 5, 15], [8, 13, 15], [6, 10, 14], [3, 7, 13], [3, 4, 5], [0, 2, 9], [5, 6, 7], [8, 15, 17], [0, 11, 16], [5, 7, 15], [7, 12, 15], [6, 8, 9], [0, 1, 12], [6, 8, 11], [0, 1, 7], [0, 5, 8], [0, 13, 15], [0, 3, 11], [0, 4, 17], [9, 15, 17], [5, 11, 16], [9, 16, 17], [4, 11, 15], [8, 12, 14], [3, 5, 16], [3, 5, 8], [0, 6, 16], [2, 3, 14], [4, 6, 12], [4, 12, 16], [6, 8, 13], [5, 9, 17], [5, 9, 11], [3, 7, 14], [8, 9, 12], [0, 5, 17], [4, 7, 10], [3, 9, 14], [3, 9, 17], [1, 2, 4], [11, 14, 16], [0, 2, 10], [1, 3, 6], [2, 8, 15], [5, 8, 16], [3, 4, 9], [2, 8, 14], [2, 5, 9], [1, 4, 5], [4, 6, 13], [9, 12, 17], [7, 12, 17], [3, 6, 16], [10, 13, 14], [5, 10, 17], [6, 11, 17], [5, 12, 15], [4, 12, 14], [2, 7, 17], [3, 12, 14], [0, 3, 15], [1, 4, 16], [6, 11, 15], [0, 4, 9], [0, 10, 14], [0, 10, 13], [4, 9, 16], [0, 6, 8], [0, 4, 14], [4, 9, 14], [2, 14, 16], [13, 14, 17], [0, 3, 9], [2, 8, 12], [7, 16, 17], [6, 14, 15], [3, 13, 15], [4, 10, 16], [9, 11, 15], [4, 7, 9], [4, 10, 11], [1, 2, 5], [6, 7, 8], [0, 3, 17], [3, 9, 15], [0, 13, 17], [3, 12, 15], [5, 6, 17], [4, 5, 13], [6, 10, 17], [1, 5, 10], [1, 2, 6], [6, 12, 14], [11, 13, 14], [1, 11, 17], [0, 6, 14], [0, 2, 8], [1, 2, 7], [1, 6, 8], [1, 3, 12], [3, 4, 11], [7, 12, 13], [12, 13, 14], [4, 7, 13], [2, 5, 14], [1, 2, 3], [7, 8, 10], [0, 4, 12], [0, 4, 6], [2, 3, 4], [8, 14, 16], [1, 3, 16], [1, 15, 16], [3, 15, 16], [0, 5, 13], [1, 9, 16], [6, 9, 15], [5, 9, 16], [1, 5, 14], [7, 13, 16], [1, 4, 15], [10, 11, 17], [1, 3, 4], [2, 12, 16], [0, 11, 17], [4, 9, 12], [5, 13, 17], [2, 6, 13], [4, 9, 15], [5, 7, 8], [8, 11, 13], [1, 9, 15], [9, 10, 16], [0, 11, 15], [0, 2, 6], [0, 1, 17], [1, 6, 17], [9, 11, 13], [2, 9, 12], [4, 7, 8], [1, 2, 12], [1, 11, 16], [4, 11, 13], [1, 11, 12], [4, 8, 15], [2, 3, 15], [2, 15, 16], [2, 10, 13], [0, 5, 12], [13, 15, 17], [3, 4, 13], [5, 16, 17], [9, 11, 12], [2, 10, 11], [3, 4, 14], [6, 10, 16], [8, 10, 15], [4, 5, 16], [4, 6, 10], [2, 3, 6], [8, 9, 14], [9, 12, 14], [5, 10, 11], [12, 14, 16], [0, 8, 9], [6, 16, 17], [4, 6, 9], [15, 16, 17], [7, 12, 14], [3, 5, 9], [1, 7, 12], [0, 8, 17], [0, 5, 11], [12, 13, 17], [6, 12, 16], [5, 10, 14], [1, 3, 11], [3, 9, 11], [7, 8, 16], [1, 4, 13], [1, 6, 9], [6, 9, 17], [8, 15, 16], [1, 3, 7], [0, 1, 5], [6, 12, 17], [3, 4, 17], [8, 11, 16], [11, 16, 17], [3, 8, 9], [7, 13, 17], [1, 6, 12], [1, 2, 13], [4, 13, 17], [0, 4, 11], [2, 6, 9], [2, 6, 11], [5, 6, 10], [7, 8, 9], [2, 3, 10], [0, 4, 13], [3, 6, 10], [6, 8, 14], [2, 7, 11], [4, 7, 12], [10, 13, 15], [1, 5, 12], [5, 14, 17], [2, 4, 7], [1, 4, 17], [4, 14, 15], [3, 8, 16], [12, 15, 16], [7, 8, 14], [2, 4, 12], [9, 14, 17], [3, 8, 14], [1, 4, 11], [0, 5, 9], [4, 8, 17], [1, 11, 15], [1, 4, 9], [0, 3, 16], [4, 8, 16], [0, 3, 6], [3, 8, 11], [4, 10, 17], [1, 7, 17], [8, 9, 10], [3, 4, 12], [2, 8, 9], [5, 11, 12], [1, 12, 16], [3, 9, 10], [2, 7, 8], [1, 3, 14], [4, 6, 16], [2, 5, 16], [1, 6, 11], [2, 13, 15], [5, 10, 16], [14, 15, 17], [0, 2, 15], [3, 6, 15], [2, 12, 13], [11, 14, 17], [2, 6, 7], [1, 2, 14], [1, 5, 6], [7, 10, 15], [7, 11, 12], [0, 6, 15], [3, 6, 7], [1, 11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70 second deadline with 10s safety margin
    
    # Preprocess: convert triples to frozensets and build index
    triples = P['edge_order']
    n = P['n']
    
    # Map each triple (as frozenset) to its 1-based index
    triple_to_idx = {}
    for i, t in enumerate(triples):
        fs = frozenset(t)
        if fs not in triple_to_idx:
            triple_to_idx[fs] = i + 1
    
    # Build adjacency: for each vertex, list of (triple_index, frozenset) containing it
    vertex_triples = [[] for _ in range(n)]
    for i, t in enumerate(triples):
        fs = frozenset(t)
        idx = triple_to_idx[fs]
        for v in t:
            vertex_triples[v].append((idx, fs))
    
    # Sort each vertex's triples by index (earliest first)
    for v in range(n):
        vertex_triples[v].sort(key=lambda x: x[0])
    
    # Default fallback: the seed solution
    default_result = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]]
    default_score = max(triple_to_idx.get(frozenset(t), float('inf')) for t in default_result)
    
    best_result = default_result
    best_score = default_score
    
    # Try to find a better matching using backtracking
    # We want to minimize the maximum 1-based index among the 6 triples
    
    def backtrack(used_vertices, current_triples, max_idx_so_far, limit):
        """
        Try to find 6 disjoint triples all with index <= limit.
        used_vertices: set of vertices already used
        current_triples: list of triples chosen so far
        max_idx_so_far: max index of triples chosen so far
        limit: we only consider triples with index <= limit
        Returns True if found, and updates best_result/best_score if successful
        """
        nonlocal best_result, best_score
        
        if time.time() > deadline:
            return False
        
        if len(current_triples) == 6:
            # Found a valid matching
            if max_idx_so_far < best_score:
                best_score = max_idx_so_far
                best_result = [list(t) for t in current_triples]
            return True
        
        # Find the unused vertex with the fewest available triples (most constrained)
        unused = [v for v in range(n) if v not in used_vertices]
        if not unused:
            return len(current_triples) == 6
        
        # Pick the vertex with fewest options under the limit
        best_v = -1
        best_count = float('inf')
        best_options = None
        for v in unused:
            options = [(idx, fs) for idx, fs in vertex_triples[v] if idx <= limit]
            if len(options) < best_count:
                best_count = len(options)
                best_v = v
                best_options = options
            if best_count == 0:
                return False
        
        if best_count == 0:
            return False
        
        # Try each option for the most constrained vertex
        for idx, fs in best_options:
            if idx > limit:
                continue
            new_used = used_vertices | fs
            new_max = max(max_idx_so_far, idx)
            if backtrack(new_used, current_triples + [list(fs)], new_max, limit):
                return True
        
        return False
    
    # Binary search on the limit: find the smallest L such that a matching exists with all indices <= L
    # First, check if the default works (it should)
    # Try smaller limits first
    
    # Get all unique indices
    all_indices = sorted(set(triple_to_idx.values()))
    
    # Binary search for the minimum possible max index
    lo = 1
    hi = max(all_indices) if all_indices else 1
    
    # First, try to find if there's a matching with a small max index
    # We'll do a binary search
    found = False
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        # Try to find a matching with all triples having index <= mid
        used = set()
        result = []
        
        def bt(used, result, limit):
            nonlocal best_result, best_score
            if time.time() > deadline:
                return False
            if len(result) == 6:
                max_idx = max(triple_to_idx[frozenset(t)] for t in result)
                if max_idx < best_score:
                    best_score = max_idx
                    best_result = [list(t) for t in result]
                return True
            
            unused = [v for v in range(n) if v not in used]
            if not unused:
                return len(result) == 6
            
            # Most constrained vertex
            best_v = -1
            best_count = float('inf')
            best_opts = None
            for v in unused:
                opts = [(idx, fs) for idx, fs in vertex_triples[v] if idx <= limit]
                if len(opts) < best_count:
                    best_count = len(opts)
                    best_v = v
                    best_opts = opts
                if best_count == 0:
                    return False
            
            if best_count == 0:
                return False
            
            for idx, fs in best_opts:
                new_used = used | fs
                if bt(new_used, result + [list(fs)], limit):
                    return True
            return False
        
        if bt(set(), [], mid):
            found = True
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Also try greedy approaches for additional exploration
    if time.time() < deadline - 5:
        # Greedy: pick triples in order of index, skip if conflicts
        def greedy():
            nonlocal best_result, best_score
            used = set()
            result = []
            for i, t in enumerate(triples):
                fs = frozenset(t)
                idx = triple_to_idx[fs]
                if fs.isdisjoint(used):
                    result.append(list(t))
                    used = used | fs
                    if len(result) == 6:
                        max_idx = max(triple_to_idx[frozenset(r)] for r in result)
                        if max_idx < best_score:
                            best_score = max_idx
                            best_result = [list(r) for r in result]
                        return
            return
        
        greedy()
    
    # Randomized greedy for additional exploration
    if time.time() < deadline - 3:
        rng = random.Random(42)
        for attempt in range(100):
            if time.time() > deadline - 2:
                break
            used = set()
            result = []
            # Shuffle triples but bias toward earlier ones
            indices = list(range(len(triples)))
            rng.shuffle(indices)
            # Sort by a mix of original index and random
            shuffled = sorted(indices, key=lambda i: triples[i] and (triple_to_idx[frozenset(triples[i])] * 0.1 + rng.random()))
            for i in shuffled:
                fs = frozenset(triples[i])
                idx = triple_to_idx[fs]
                if fs.isdisjoint(used):
                    result.append(list(triples[i]))
                    used = used | fs
                    if len(result) == 6:
                        max_idx = max(triple_to_idx[frozenset(r)] for r in result)
                        if max_idx < best_score:
                            best_score = max_idx
                            best_result = [list(r) for r in result]
                        break
    
    # Ensure we return valid types
    final_result = []
    for t in best_result:
        final_result.append([int(v) for v in t])
    
    return final_result

# EVOLVE-BLOCK-END
