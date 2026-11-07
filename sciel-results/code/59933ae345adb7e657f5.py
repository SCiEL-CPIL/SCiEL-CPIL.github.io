"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Maximum clique via branch and bound
    # Use a simple but effective approach: order vertices by degree, use greedy coloring bound
    
    # Vertex ordering by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # Map original vertex to position in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Rebuild adjacency in ordered form
    adj_ordered = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            adj_ordered[pos[u]].add(pos[v])
    
    # Convert to sorted lists for efficiency
    adj_lists = [sorted(adj_ordered[i]) for i in range(n)]
    
    best_clique = []
    
    # Greedy coloring for upper bound
    def greedy_color(candidates, adj_set):
        """Color the candidates greedily to get an upper bound on clique size."""
        color = [0] * len(candidates)
        color_class = []
        uncolored = list(range(len(candidates)))
        max_color = 0
        while uncolored:
            max_color += 1
            color_class.append([])
            new_uncolored = []
            for idx in uncolored:
                # Check if candidate can take this color
                can_color = True
                for prev in color_class[-1]:
                    if candidates[idx] in adj_set[candidates[prev]]:
                        # They are adjacent, so they can't share a color in this context
                        # Actually for coloring bound, we color so that non-adjacent get same color
                        # Wait, we want: if two vertices are NOT adjacent, they can share a color
                        # So we can assign color c to vertex v if no vertex already colored c is adjacent to v
                        pass
                # Simpler: just do a basic greedy
                pass
            
            # Let me redo this properly
            break
        return max_color
    
    # Simpler approach: just use branch and bound with a basic bound
    def max_clique(candidates, current_clique, adj_sets):
        """Branch and bound for maximum clique."""
        nonlocal best_clique
        if not candidates:
            if len(current_clique) > len(best_clique):
                best_clique = list(current_clique)
            return
        
        # Upper bound: len(current_clique) + len(candidates)
        if len(current_clique) + len(candidates) <= len(best_clique):
            return
        
        # Try vertices in order of decreasing degree within candidates
        for i in range(len(candidates) - 1, -1, -1):
            v = candidates[i]
            new_clique = current_clique + [v]
            # New candidates are those in candidates[0:i] that are adjacent to v
            new_candidates = [c for c in candidates[:i] if c in adj_sets[v]]
            max_clique(new_candidates, new_clique, adj_sets)
    
    # Use original vertex indices
    adj_sets = [set() for _ in range(n)]
    for u, v in edges:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    max_clique(list(range(n)), [], adj_sets)
    
    # Convert best_clique back to original vertex indices
    # Wait, I used original indices in the last call, so best_clique is already in original indices
    
    if len(best_clique) < 2:
        # Fallback
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branches = [int(v) for v in best_clique]
    
    # Build paths: for each pair, use the direct edge
    from itertools import combinations
    paths = []
    for u, v in combinations(branches, 2):
        # Verify edge exists
        if v in adj_sets[u]:
            paths.append([int(u), int(v)])
        else:
            # Should not happen for a clique
            paths.append([int(u), int(v)])
    
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
