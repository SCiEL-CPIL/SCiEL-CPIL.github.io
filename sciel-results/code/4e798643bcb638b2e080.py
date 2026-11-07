"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build initial adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all valid candidate edges (non-base, and initially triangle-free)
    # An edge (u,v) is valid if u and v don't share a common neighbor in current adj
    # We'll check dynamically during search
    
    # Generate all non-base pairs
    all_pairs = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                all_pairs.append((u, v))
    
    # Sort by path distance descending (longer jumps first)
    all_pairs.sort(key=lambda p: p[1]-p[0], reverse=True)
    
    def creates_triangle(u, v, adj):
        return len(adj[u] & adj[v]) > 0
    
    def compute_all_distances(adj, start):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        return dist
    
    def check_diameter(adj, max_diam=4):
        for start in range(n):
            dist = compute_all_distances(adj, start)
            if any(d == -1 for d in dist):
                return False
            if any(d > max_diam for d in dist):
                return False
        return True
    
    def compute_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = compute_all_distances(adj, start)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    # IDDFS: try to find solution with exactly k edges
    # Lower bound: we need at least some edges. Start from 3 and go up.
    # For a path of 24 vertices, diameter is 23. Each edge can reduce diameter.
    # With k shortcut edges, we can create a structure with diameter ~24/(k+1) roughly.
    # For diameter 4, we need roughly 24/4 - 1 = 5 edges minimum, but let's start lower.
    
    best_solution = None
    
    for max_edges in range(3, 12):
        solution = []
        
        def dfs(current_edges, current_adj, count, limit):
            if count == limit:
                if check_diameter(current_adj):
                    return True
                return False
            
            if count > limit:
                return False
            
            # Pruning: if even adding all remaining edges can't reduce diameter enough
            # Simple pruning: check current diameter
            curr_diam = compute_diameter(current_adj)
            if curr_diam <= 4:
                # We can stop early if we already have diameter <= 4 with fewer edges
                # But we're searching for exactly 'limit' edges, so this means we found a solution
                # with fewer edges. Actually, if curr_diam <= 4, we can return True
                # because we have a valid solution with 'count' edges <= limit
                return True
            
            # Try each candidate edge
            for idx, (u, v) in enumerate(all_pairs):
                if count + 1 > limit:
                    break
                
                if creates_triangle(u, v, current_adj):
                    continue
                
                # Add edge
                current_edges.append((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
                if dfs(current_edges, current_adj, count + 1, limit):
                    return True
                
                # Remove edge
                current_edges.pop()
                current_adj[u].remove(v)
                current_adj[v].remove(u)
            
            return False
        
        adj_copy = [set(s) for s in adj]
        if dfs(solution, adj_copy, 0, max_edges):
            best_solution = list(solution)
            break
    
    if best_solution:
        return [[u, v] for u, v in best_solution]
    
    # Fallback: greedy approach
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    added_edges = []
    remaining = set(all_pairs)
    
    while True:
        diam = compute_diameter(adj)
        if diam <= 4:
            break
        if not remaining:
            break
        
        best_edge = None
        best_new_diam = float('inf')
        
        for (u, v) in remaining:
            if creates_triangle(u, v, adj):
                continue
            
            adj[u].add(v)
            adj[v].add(u)
            new_diam = compute_diameter(adj)
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append((u, v))
        remaining.discard((u, v))
    
    if compute_diameter(adj) <= 4:
        return [[u, v] for u, v in added_edges]
    
    return [[u, v] for u, v in added_edges]

# EVOLVE-BLOCK-END
