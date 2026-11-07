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
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: all pairs not in base, not adjacent (distance >= 3 in path to avoid immediate triangles)
    # Actually, we need to check dynamically. Let's generate all non-base pairs.
    all_pairs = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                all_pairs.append((u, v))
    
    # Check if adding edge (u,v) to current adj creates a triangle
    def creates_triangle(u, v, adj):
        # Check if u and v share a common neighbor
        return len(adj[u] & adj[v]) > 0
    
    # BFS to check diameter
    def check_diameter(adj, max_diam=4):
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_diam:
                            return False
                        q.append(nb)
            if any(d == -1 for d in dist):
                return False
        return True
    
    # Greedy approach: try to add edges that most reduce diameter
    # Use a heuristic: prioritize edges that connect distant vertices
    
    # Let's try a direct construction approach
    # For a path of length 23, to get diameter 4, we need shortcuts
    
    # Strategy: add edges in a pattern that ensures diameter <= 4
    # One known approach: add edges (i, i+k) for appropriate k
    
    # Let me try: add edges to create a structure where from any vertex, 
    # you can reach any other in 4 steps
    
    # Attempt: add edges (0,5), (1,6), (2,7), ..., but check triangle-free
    # Actually, let's use a smarter approach: binary search / optimization
    
    # Let me try a specific construction:
    # Add edges to make the graph "jump" across the path
    
    # For diameter 4 with 24 vertices, we need at least some edges.
    # Let's try adding edges: (0,6), (1,7), (2,8), (3,9), (4,10), (5,11), (6,12), (7,13), (8,14), (9,15), (10,16), (11,17), (12,18), (13,19), (14,20), (15,21), (16,22), (17,23)
    # But this might create triangles
    
    # Better: use a backtracking search with pruning
    
    # Sort candidates by "usefulness" - distance in path
    all_pairs_sorted = sorted(all_pairs, key=lambda p: p[1]-p[0], reverse=True)
    
    # Backtracking
    best_solution = None
    best_count = float('inf')
    
    def backtrack(current_edges, current_adj, idx, count):
        nonlocal best_solution, best_count
        
        if count >= best_count:
            return
        
        # Check if current state already has diameter <= 4
        if check_diameter(current_adj):
            if count < best_count:
                best_count = count
                best_solution = list(current_edges)
            return
        
        if idx >= len(all_pairs_sorted):
            return
        
        # Try adding edge idx
        u, v = all_pairs_sorted[idx]
        
        if not creates_triangle(u, v, current_adj):
            # Add edge
            current_edges.append((u, v))
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            backtrack(current_edges, current_adj, idx+1, count+1)
            
            # Remove edge
            current_edges.pop()
            current_adj[u].remove(v)
            current_adj[v].remove(u)
        
        # Try skipping edge idx
        backtrack(current_edges, current_adj, idx+1, count)
    
    # This full backtracking is too slow. Let's use a smarter approach.
    # Greedy with local search
    
    # Greedy: repeatedly add the edge that most reduces the maximum distance
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    added_edges = []
    
    # Compute current diameter
    def compute_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        q.append(nb)
        return max_d
    
    # Greedy: add edges that reduce diameter most
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
            
            # Try adding
            adj[u].add(v)
            adj[v].add(u)
            new_diam = compute_diameter(adj)
            
            # Undo
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
    
    # Verify
    if compute_diameter(adj) <= 4:
        return [[u, v] for u, v in added_edges]
    
    # Fallback: try a different construction
    # Let's try a known good pattern
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Add edges: (i, i+5) for i in 0..18, but check triangle-free
    candidate_edges = []
    for i in range(n):
        for j in range(i+3, n):  # distance at least 3 to avoid path triangles
            if (i, j) not in base_edges and not creates_triangle(i, j, adj):
                candidate_edges.append((i, j))
    
    # Try a specific set
    test_edges = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15), (11, 16), (12, 17), (13, 18), (14, 19), (15, 20), (16, 21), (17, 22), (18, 23)]
    
    adj2 = [set() for _ in range(n)]
    for u, v in base_edges:
        adj2[u].add(v)
        adj2[v].add(u)
    
    valid_edges = []
    for u, v in test_edges:
        if not creates_triangle(u, v, adj2):
            adj2[u].add(v)
            adj2[v].add(u)
            valid_edges.append((u, v))
    
    if compute_diameter(adj2) <= 4:
        return [[u, v] for u, v in valid_edges]
    
    # Last resort: return greedy result even if not perfect (shouldn't happen)
    return [[u, v] for u, v in added_edges]

# EVOLVE-BLOCK-END
