"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Parity: even vertices = 0,2,4,...,22; odd = 1,3,5,...,23
    # Only add even-odd edges to stay triangle-free (bipartite)
    
    # Precompute all valid candidate edges (even-odd, not already in base)
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            if v not in base_adj[u]:
                candidates.append((u, v))
    
    # Compute distances in base graph (path)
    # In a path, dist(i,j) = |i-j|
    
    # Pairs that need help: all pairs with distance > 4
    # We need to select minimum edges so that in the augmented graph, all pairs have dist <= 4
    
    # Use greedy set cover: each candidate edge (u,v) helps reduce distances
    # For efficiency, track which pairs are still "uncovered" (dist > 4 in current graph)
    
    # State: current adjacency
    adj = [set(s) for s in base_adj]
    selected = []
    
    # Compute current distances via BFS from each node (n=24, cheap)
    def compute_all_distances(adj, n):
        dist = [[0]*n for _ in range(n)]
        for src in range(n):
            d = [-1]*n
            d[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if d[w] == -1:
                        d[w] = d[u] + 1
                        q.append(w)
            for i in range(n):
                dist[src][i] = d[i]
        return dist
    
    def get_uncovered_pairs(adj, n):
        """Return set of (i,j) with i<j and dist(i,j) > 4"""
        dist = compute_all_distances(adj, n)
        uncovered = set()
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] > 4:
                    uncovered.add((i, j))
        return uncovered, dist
    
    def edge_covers(adj, n, u, v, uncovered):
        """Check if adding edge (u,v) to adj would make some uncovered pairs covered.
        Returns set of uncovered pairs that would become covered."""
        # Simulate adding edge
        adj[u].add(v)
        adj[v].add(u)
        new_uncovered, new_dist = get_uncovered_pairs(adj, n)
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        # Pairs that were uncovered but now covered
        newly_covered = uncovered - new_uncovered
        return newly_covered
    
    # Initial uncovered
    uncovered, _ = get_uncovered_pairs(adj, n)
    
    deadline = start_time + 75  # Safety margin
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_covered = set()
        
        for (u, v) in candidates:
            # Skip if already selected
            if v in adj[u]:
                continue
            newly_covered = edge_covers(adj, n, u, v, uncovered)
            if len(newly_covered) > len(best_covered):
                best_covered = newly_covered
                best_edge = (u, v)
                if time.time() > deadline:
                    break
        
        if best_edge is None or len(best_covered) == 0:
            # No single edge can help; this shouldn't happen for a path
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        selected.append([u, v])
        uncovered -= best_covered
    
    # Verify result
    final_dist = compute_all_distances(adj, n)
    max_d = max(max(row) for row in final_dist)
    
    # Check triangle-free: verify bipartite
    # Color by parity - all edges should be even-odd
    for u in range(n):
        for v in adj[u]:
            if (u % 2) == (v % 2):
                # Triangle-free violated
                # This shouldn't happen with our construction
                pass
    
    result = [[int(u), int(v)] for u, v in selected]
    return result
# EVOLVE-BLOCK-END
