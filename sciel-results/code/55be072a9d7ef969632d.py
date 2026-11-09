"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    """Find a large clique subdivision in the pinned graph."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P_set, X):
        """Find maximum clique in graph."""
        if not P_set and not X:
            return list(R)
        
        # Pivot selection: choose vertex with max degree in P ∪ X
        candidates = P_set | X
        if not candidates:
            return list(R)
        
        # Pick pivot with max degree in P
        pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
        
        # Only consider non-neighbors of pivot
        for v in list(P_set - adj[pivot]):
            result = bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            if result and len(result) > len(max_clique):
                max_clique.extend(result)
                max_clique.clear()
                max_clique.extend(result)
            P_set.remove(v)
            X.add(v)
        
        return list(R)
    
    # Find maximum clique
    max_clique = []
    bron_kerbosch(set(), set(range(n)), set())
    
    # Fallback if no clique found
    if len(max_clique) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Try to find larger clique subdivisions
    # A clique subdivision of size k has k branch vertices
    # The branch vertices form a clique, and between each pair there's a path
    # with disjoint interiors
    
    # Start with the maximum clique
    branches = list(max_clique)
    
    # Try to add more branch vertices by finding vertices that can connect
    # to existing branches with disjoint paths
    
    # For simplicity, try to find a larger clique first
    # If the maximum clique is small, try to find a subdivision
    
    # Check if we can find a larger structure
    # For now, return the maximum clique as branches with direct edges as paths
    
    # Verify the clique
    clique_set = set(branches)
    valid = True
    for u, v in combinations(branches, 2):
        if v not in adj[u]:
            valid = False
            break
    
    if not valid:
        # Fallback to a simple edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Build paths: direct edges for clique members
    paths = []
    for u, v in combinations(branches, 2):
        paths.append([int(u), int(v)])
    
    # Convert to proper types
    branches_out = [int(b) for b in branches]
    paths_out = [[int(x) for x in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
