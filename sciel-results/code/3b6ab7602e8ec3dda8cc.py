"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    # Total edges
    m = len(edge_set)
    
    if m == 0:
        return []
    
    # Find all maximal bicliques using a heuristic approach
    # For each pair of vertices (u, v) that form an edge, try to extend to a maximal biclique
    # A biclique is defined by (L, R) where L and R are disjoint, and every u in L is connected to every v in R
    
    # Strategy: enumerate candidate bicliques by growing from seed edges
    # To limit the search, we use a greedy + local improvement approach combined with exact cover
    
    # First, let's find a good set of candidate bicliques
    # We'll use the following heuristic:
    # For each vertex u, consider its neighbors. Partition neighbors into groups that form bicliques.
    
    # Better approach: find all "maximal" bicliques via a limited enumeration
    # Given the graph is small (30 vertices), we can try to find good bicliques
    
    # Let's use a different strategy: 
    # 1. Find a greedy biclique cover
    # 2. Try to improve it by merging/optimizing
    
    # Greedy approach: repeatedly find the largest biclique, add it, remove its edges
    def find_largest_biclique(remaining_edges, adj_remaining):
        """Find a large biclique in the remaining graph using a heuristic."""
        if not remaining_edges:
            return None, None, set()
        
        # Try each edge as a seed
        best_size = 0
        best_biclique = None
        
        # Limit the number of seeds we try
        edge_list = list(remaining_edges)
        # Sort by degree product to try promising ones first
        for u, v in edge_list:
            # Start with biclique ({u}, {v})
            L = {u}
            R = {v}
            
            # Try to grow the biclique
            changed = True
            while changed:
                changed = False
                # Try adding a vertex to L
                for x in range(n):
                    if x not in L and x not in R:
                        # x can join L if connected to all in R
                        if all(x in adj_remaining[y] for y in R):
                            L.add(x)
                            changed = True
                # Try adding a vertex to R
                for x in range(n):
                    if x not in L and x not in R:
                        # x can join R if connected to all in L
                        if all(x in adj_remaining[y] for y in L):
                            R.add(x)
                            changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None:
            return None, None, set()
        
        L, R = best_biclique
        # Get the edges in this biclique
        biclique_edges = set()
        for u in L:
            for v in R:
                e = frozenset((u, v))
                if e in remaining_edges:
                    biclique_edges.add(e)
        
        return L, R, biclique_edges
    
    # Greedy cover
    remaining = set(edge_set)
    adj_remaining = [set(adj[i]) for i in range(n)]
    cover = []
    
    while remaining:
        L, R, biclique_edges = find_largest_biclique(remaining, adj_remaining)
        if L is None or not biclique_edges:
            # Fallback: pick any edge
            u, v = next(iter(remaining))
            L = {u}
            R = {v}
            biclique_edges = {frozenset((u, v))}
        
        cover.append((L, R))
        for e in biclique_edges:
            remaining.discard(e)
            u, v = tuple(e)
            if u < v:
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            else:
                adj_remaining[v].discard(u)
                adj_remaining[u].discard(v)
    
    num_bicliques = len(cover)
    
    # Now try to improve: can we merge some bicliques or find a better cover?
    # Let's try a local search: for each biclique, try to see if it can be replaced
    # by a larger one that covers more edges
    
    # Actually, let's try a different approach: use the greedy result as a starting point
    # and try to reduce the number of bicliques
    
    # For now, let's also try a simpler heuristic: 
    # Group edges by one endpoint and see if we can form bicliques that way
    
    # Let's try to find a better cover using a different heuristic
    # For each vertex, look at its neighbors and try to partition them
    
    # Alternative: use the fact that within a biclique, all left vertices share the same neighbors on the right
    
    # Let's try a smarter greedy: instead of largest biclique, try to maximize edge coverage per biclique
    # while keeping bicliques large
    
    # For now, return the greedy cover
    # Convert to the required format: list of [left_vertices, right_vertices]
    result = []
    for L, R in cover:
        left = sorted(list(L))
        right = sorted(list(R))
        # Ensure non-empty and disjoint
        if left and right and not set(left) & set(right):
            result.append([left, right])
        else:
            # This shouldn't happen, but handle it
            result.append([[min(L)], [max(R)]])
    
    return result
# EVOLVE-BLOCK-END
