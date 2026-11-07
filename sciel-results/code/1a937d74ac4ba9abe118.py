"""Rewritten program for triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0  # seconds, safety margin before 90s kill

    n = P['n']
    
    # Build initial adjacency from path
    base_adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        base_adj[u][v] = True
        base_adj[v][u] = True

    def bfs_diameter(adj):
        """Compute diameter using BFS from each node. Returns (diam, dist_matrix)."""
        max_diam = 0
        dist_matrix = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            dist_matrix.append(dist)
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam, dist_matrix

    def is_triangle_free_with_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        return True

    def check_full_validity(added_edges):
        """Full validation: no dupes, no existing edges, triangle-free, connected, diam<=4."""
        edge_set = set()
        for u, v in added_edges:
            if u == v:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if base_adj[a][b]:
                return False

        adj = [row[:] for row in base_adj]
        for u, v in added_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check triangle-free
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        
        diam, _ = bfs_diameter(adj)
        return diam <= 4

    # Hardcoded known-good 8-edge solution as fallback
    hardcoded = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
    
    if check_full_validity(hardcoded):
        best_solution = hardcoded
    else:
        best_solution = None

    # Strategy: Try to find solutions with fewer edges (5, 6, 7)
    # Key structural insight for triangle-free graphs:
    # - If vertex v has neighbors N(v), then N(v) is an independent set
    # - For diameter 4, we need "hub" vertices that reach far parts of the path
    # - In a path, the natural hubs are central vertices
    
    # Generate candidate edges that are triangle-free with base graph
    # An edge (u,v) is valid if no common neighbor exists in base graph
    # In a path, u and v have a common neighbor only if |u-v| == 2
    
    def get_valid_candidates(base_adj):
        """Get all non-path edges that don't create triangles with base graph."""
        cands = []
        for u in range(n):
            for v in range(u + 1, n):
                if not base_adj[u][v]:
                    # Check triangle-free with base: no w adjacent to both u and v in base
                    conflict = False
                    for w in range(n):
                        if w != u and w != v and base_adj[u][w] and base_adj[v][w]:
                            conflict = True
                            break
                    if not conflict:
                        cands.append((u, v))
        return cands

    # For iterative deepening, we need efficient checking
    # Precompute base distances
    base_diam, base_dist = bfs_diameter(base_adj)
    
    # Try edge counts from 5 upward
    # For small k, we can do smarter search
    # Key: in triangle-free graph, if we add edges from hub h to set S,
    # then S must be an independent set in the resulting graph
    
    # Approach: Enumerate hub vertices and their neighbor sets
    # A hub at position h can connect to vertices at distances >= 3 (to avoid triangles)
    # In path, vertex h is adjacent to h-1, h+1. So h can connect to any v where |v-h| >= 3
    
    # For diameter 4 with k edges, we need the graph to be "compact"
    # Let's try a targeted search: pick 1-3 hub vertices, assign edges
    
    # First, try single-hub solutions
    # A single hub h with edges to S (|S|=k) gives distances:
    # dist(i,j) = min(|i-j|, dist(i,h)+dist(h,j)) where dist via h = (1 if i in S or i==h else inf) + ...
    # Actually dist(i,h) = 0 if i==h, 1 if i in S, else |i-h| (via path)
    # But wait, if i is not in S and i != h, dist(i,h) might be shortened by other S vertices
    
    # Let me just do a direct search for k=5,6,7
    # For k edges, the number of ways to choose from ~276 candidates is C(276,5) ~ 10^10 - too many
    # Need smarter approach
    
    # Better: use the fact that in triangle-free graph, edges from a hub must go to independent set
    # And for diameter 4, we need coverage of the path
    
    # Let's try: for each potential hub (or small set of hubs), find minimum edges
    # to achieve diameter 4
    
    # Simpler structural approach:
    # In a triangle-free graph with diameter 4 on a path, 
    # the "useful" edges are those connecting vertices far apart on the path
    # Let's focus on edges with |u-v| >= 5 (shorter edges give less diameter reduction)
    
    # Actually, let me try a different structural approach:
    # Model as: we need to select edges so that for all pairs (i,j) with |i-j| > 4,
    # there's a path of length <= 4.
    
    # A path of length <= 4 from i to j can use at most 4 edges.
    # If we use t added edges, the path has t added + (4-t) path edges.
    
    # For the search, let's use a heuristic: 
    # 1. Identify "critical pairs" - pairs with distance > 4
    # 2. Each added edge (u,v) can help pairs (i,j) where a shortest path through (u,v) is <= 4
    # 3. Use greedy but with better candidate ordering
    
    # Let me try yet another approach: local search / hill climbing
    # Start with the 8-edge solution, try to remove edges and see if still valid
    
    if best_solution:
        # Try to reduce the 8-edge solution
        current = list(best_solution)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(len(current)):
                # Try removing edge at idx
                trial = current[:idx] + current[idx+1:]
                if check_full_validity(trial):
                    current = trial
                    improved = True
                    break
        
        if len(current) < len(best_solution):
            best_solution = current

    # Now try to find even better solutions via targeted search
    # Try 2-hub configurations
    # For two hubs h1, h2 with edges to S1, S2:
    # Need S1 independent, S2 independent, and no triangles across
    
    # Let's try a more systematic approach for k=5,6,7
    # Use the observation that optimal solutions likely use 1-3 hub vertices
    
    # For a single hub h, the edges must go to an independent set in the path
    # In a path, an independent set at distance >= 3 from h:
    # vertices h+3, h+5, h+7, ... and h-3, h-5, h-7, ...
    # Plus we need to cover both sides for diameter 4
    
    # Let me try: for each hub h, find minimum edges to achieve diam<=4
    # This is a covering problem: each edge (h,v) covers pairs that can reach via h
    
    # For hub h with edges to S:
    # dist(i,j) <= min(|i-j|, d(i,h)+d(h,j))
    # where d(i,h) = 0 if i==h, 1 if i in S, else min over path (could be shortened)
    # Actually in the augmented graph, d(i,h) could be 1 if i in S, or 2 if i is neighbor of some s in S
    # This gets complex. Let me just brute-force small hub configurations.
    
    # For k=5: try all single-hub configurations
    # Hub h can connect to vertices at distance >= 3
    # For diameter 4, hub h needs to reach vertices within distance 4
    # So vertices at distance > 4 from h need help
    
    # Let me try a direct enumeration for very small k
    # For k=5, try all combinations of 5 "long" edges (|u-v|>=5)
    # But C(~100, 5) is still too large
    
    # Alternative: use the greedy approach but start from scratch with better heuristics
    # and also try to improve the result
    
    # Let me try: greedy with multiple restarts and different orderings
    def greedy_solve(order_hint=None):
        """Greedy set cover with optional ordering hint."""
        adj = [row[:] for row in base_adj]
        diam, dist = bfs_diameter(adj)
        if diam <= 4:
            return []
        
        # Get uncovered pairs
        def get_uncovered(dist, target=4):
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i][j] > target:
                        pairs.append((i, j))
            return set(pairs)
        
        uncovered = get_uncovered(dist)
        selected = []
        
        while uncovered and time.time() < deadline - 5:
            best_edge = None
            best_gain = 0
            
            # Generate candidates on the fly
            cands = []
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v] and is_triangle_free_with_edge(adj, u, v):
                        cands.append((u, v))
            
            # Sort by heuristic if order_hint given
            if order_hint:
                cands.sort(key=order_hint)
            
            for u, v in cands:
                # Compute how many uncovered pairs this edge helps
                gain = 0
                for i, j in uncovered:
                    # New distance via edge (u,v):
                    d1 = dist[i][u] + 1 + dist[v][j] if dist[i][u] >= 0 and dist[v][j] >= 0 else 999
                    d2 = dist[i][v] + 1 + dist[u][j] if dist[i][v] >= 0 and dist[u][j] >= 0 else 999
                    new_d = min(dist[i][j], d1, d2)
                    if new_d <= 4:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (u, v)
            
            if best_edge is None or best_gain == 0:
                break
            
            u, v = best_edge
            selected.append([u, v])
            adj[u][v] = True
            adj[v][u] = True
            diam, dist = bfs_diameter(adj)
            uncovered = get_uncovered(dist)
        
        if not uncovered:
            return selected
        return None
    
    # Try greedy with different heuristics
    # Heuristic 1: prefer longer edges (more diameter reduction)
    def edge_length(u, v):
        return -(abs(u - v))
    
    result = greedy_solve(edge_length)
    if result and len(result) < len(best_solution or []):
        if check_full_validity(result):
            best_solution = result
    
    # Heuristic 2: prefer edges from central vertices
    def centrality(u, v):
        # Prefer edges involving vertices near the center (11,12)
        score = abs(u - 11.5) + abs(v - 11.5)
        return score
    
    result = greedy_solve(centrality)
    if result and (best_solution is None or len(result) < len(best_solution)):
        if check_full_validity(result):
            best_solution = result
    
    # Try to further improve by removing edges
    if best_solution:
        current = list(best_solution)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(len(current)):
                trial = current[:idx] + current[idx+1:]
                if check_full_validity(trial):
                    current = trial
                    improved = True
                    break
        if len(current) < len(best_solution):
            best_solution = current
    
    # Final fallback
    if best_solution is None:
        if check_full_validity(hardcoded):
            best_solution = hardcoded
        else:
            return []
    
    # Ensure correct types
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
