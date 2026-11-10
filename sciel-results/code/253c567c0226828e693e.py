"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    # Build adjacency as bitmasks for fast intersection
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Convert to list of neighbor lists for the algorithm
    neighbors = [set() for _ in range(n)]
    for u, v in P['edges']:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    deadline = time.time() + 70.0  # Safety margin before 90s limit
    
    best_clique = []
    
    def greedy_color_order(candidates):
        """Order candidates by greedy coloring for upper bound."""
        # Greedy coloring: assign colors to vertices in candidates
        # The number of colors gives an upper bound on clique size in candidates
        colors = {}
        color_classes = []
        remaining = list(candidates)
        while remaining:
            color_class = []
            for v in remaining[:]:
                if all(v not in color_class for u in color_class):
                    color_class.append(v)
                    remaining.remove(v)
            color_classes.append(color_class)
        
        # Order by color (higher color = potentially in larger clique)
        ordered = []
        for cc in reversed(color_classes):
            ordered.extend(cc)
        return ordered, len(color_classes)
    
    def expand(clique, candidates):
        """Branch and bound to find max clique."""
        nonlocal best_clique, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            return
        
        # Get upper bound via greedy coloring
        ordered, num_colors = greedy_color_order(candidates)
        
        if len(clique) + num_colors <= len(best_clique):
            return
        
        # Process in reverse order (highest color first)
        for i in range(len(ordered) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = ordered[i]
            
            # Upper bound: len(clique) + i + 1
            if len(clique) + i + 1 <= len(best_clique):
                break
            
            new_clique = clique + [v]
            new_candidates = [c for c in ordered[:i] if c in neighbors[v]]
            expand(new_clique, new_candidates)
    
    # Start with all vertices
    initial_candidates = list(range(n))
    expand([], initial_candidates)
    
    # If we found a clique of size >= 2, use it
    if len(best_clique) >= 2:
        branches = [int(v) for v in best_clique]
        paths = []
        for u, v in combinations(branches, 2):
            # Direct edge path (clique guarantees adjacency)
            paths.append([int(u), int(v)])
        return {'branches': branches, 'paths': paths}
    
    # Fallback: just use an edge
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
