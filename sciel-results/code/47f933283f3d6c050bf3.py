import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set(tuple(e) for e in base_edges)
    
    # Precompute bipartition candidates: even-odd pairs only (triangle-free guarantee)
    # Partition 0: even vertices, Partition 1: odd vertices
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # Build all possible bipartite edges (excluding base path edges)
    all_candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_set and (v, u) not in base_set:
                all_candidates.append((u, v))
    
    # Precompute adjacency for base graph
    base_adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Function to compute diameter using BFS from all nodes
    def compute_diameter(adj):
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Check connectivity
            if any(d == -1 for d in dist):
                return 999
            max_diam = max(max_diam, max(dist))
        return max_diam
    
    # Fast check: is the graph triangle-free? (For bipartite edges, always true)
    # But we need to ensure no triangles are formed. Since we only add bipartite edges,
    # and the base graph is a path (bipartite), the union is always bipartite, hence triangle-free.
    # So we don't need to check triangle-free separately for bipartite candidates.
    
    # We'll use a systematic search: try edge sets of size k, starting from minimum possible.
    # Theoretical lower bound: path P24 has diameter 23. We need diameter <= 4.
    # Each edge can reduce diameter by at most 1 in best case, but realistically we need several edges.
    # Let's try k = 1, 2, 3, ... until we find a valid solution.
    
    deadline = time.time() + 85
    
    # For efficiency, we'll use a greedy heuristic to order candidates by their potential impact.
    # Sort candidates by how much they might reduce diameter (heuristic: edges connecting far-apart vertices)
    def edge_potential(u, v):
        # Heuristic: prefer edges that connect vertices far apart in the path
        return abs(u - v)
    
    all_candidates.sort(key=edge_potential, reverse=True)
    
    # Try increasing number of edges
    for k in range(1, 20):  # Upper bound on number of edges needed
        if time.time() > deadline - 5:
            break
            
        # Use a systematic search with pruning
        # For small k, we can enumerate combinations, but that's expensive.
        # Instead, use a greedy approach with backtracking for small k.
        
        # Greedy with backtracking for small k
        def search(start_idx, current_edges, current_adj):
            if time.time() > deadline - 5:
                return None
            
            if len(current_edges) == k:
                # Check if diameter <= 4
                diam = compute_diameter(current_adj)
                if diam <= 4:
                    return list(current_edges)
                return None
            
            if start_idx >= len(all_candidates):
                return None
                
            # Pruning: if we can't reach diameter 4 with remaining edges, return None
            # (This is hard to prune effectively, so we skip)
            
            # Try adding each candidate from start_idx
            for i in range(start_idx, len(all_candidates)):
                u, v = all_candidates[i]
                
                # Check if adding this edge creates a duplicate
                if v in current_adj[u]:
                    continue
                    
                # Tentatively add
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_edges.append((u, v))
                
                # Recurse
                result = search(i + 1, current_edges, current_adj)
                if result is not None:
                    return result
                    
                # Undo
                current_edges.pop()
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
            return None
        
        # Start search with base adjacency
        initial_adj = {i: set(base_adj[i]) for i in range(n)}
        result = search(0, [], initial_adj)
        
        if result is not None:
            # Convert to list of lists with int types
            return [[int(u), int(v)] for u, v in result]
        
        # If no solution found with k edges, try k+1
        # But to avoid exponential blowup, limit the search depth
    
    # Fallback: use a known valid solution if search fails
    # This is a heuristic solution that should work
    # Add edges to create a "shortcut" structure
    fallback_edges = [
        [0, 12],
        [1, 13],
        [2, 14],
        [3, 15],
        [4, 16],
        [5, 17],
        [6, 18],
        [7, 19],
        [8, 20],
        [9, 21],
        [10, 22],
        [11, 23]
    ]
    
    # Verify fallback is valid
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in fallback_edges:
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
    
    diam = compute_diameter(adj)
    if diam <= 4:
        return [[int(u), int(v)] for u, v in fallback_edges]
    
    return []

# EVOLVE-BLOCK-END
