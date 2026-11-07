"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    base_edges = [(u, v) for u, v in P['edges']]
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    # Precompute: all valid non-base edges (no triangle with path)
    # An edge (u,v) is valid if |u-v| >= 3 (no triangle with consecutive path edges)
    # and it's not a base edge
    all_candidate_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edge_set:
                continue
            if abs(v - u) <= 2:
                continue  # would create triangle
            all_candidate_edges.append((u, v))
    
    def _build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def _check_triangle_free_fast(n, adj, added_set):
        """Check triangle-free using only added edges for new triangles."""
        for (u, v) in added_set:
            # Check if u and v share a common neighbor
            common = adj[u] & adj[v]
            if common:
                return False
        return True
    
    def _bfs_distances(n, adj, src):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist
    
    def _is_valid(added_edges):
        """Check if base + added is triangle-free, connected, diameter <= 4."""
        all_e = list(base_edges) + list(added_edges)
        adj = _build_adj(n, all_e)
        
        # Triangle-free check
        added_set = set((min(u,v), max(u,v)) for u,v in added_edges)
        if not _check_triangle_free_fast(n, adj, added_set):
            return False
        
        # Diameter check
        for src in range(n):
            dist = _bfs_distances(n, adj, src)
            if -1 in dist:
                return False
            if max(dist) > 4:
                return False
        return True
    
    # Strategy: Use a smarter constructive approach.
    # On a path of length 23, we need to reduce diameter from 23 to 4.
    # Key insight: each added edge (u,v) creates a shortcut. 
    # We need enough shortcuts so that any two nodes are within distance 4.
    
    # Precompute which pairs of nodes are "far" (distance > 4 in path)
    # and need to be connected via added edges.
    
    # Try a structured approach: select hub nodes and connect them strategically.
    # The minimum number is likely around 6-8.
    
    # Let's try a different approach: model as a set cover problem.
    # Each candidate edge covers certain "far pairs" by providing a shorter path.
    
    # Actually, let's use a much faster validation by caching.
    
    def _validate_edges(added_list):
        """Fast validation using precomputed structures."""
        added_set = frozenset((min(u,v), max(u,v)) for u,v in added_list)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Triangle check: for each added edge, check no common neighbor
        for u, v in added_list:
            if adj[u] & adj[v]:
                return False
        
        # Also check triangles among added edges themselves
        added_adj = {e: set() for e in added_set}
        for u, v in added_list:
            key = (min(u,v), max(u,v))
            for w, x in added_list:
                if (u,w) in [(a,b) for a,b in base_edges] or (v,w) in [(a,b) for a,b in base_edges]:
                    pass  # handled above
            # Check if two added edges share a vertex and form triangle
        # More carefully: check all triangles
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        
        # Diameter check
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > 4:
                            return False
                        queue.append(nb)
            if -1 in dist:
                return False
        return True
    
    # Use a branch-and-bound with the candidate edges, but smarter.
    # First, find a good initial solution using greedy.
    
    # Greedy: start with no edges, add the edge that reduces the number of "far pairs" the most.
    
    def _count_far_pairs(adj):
        """Count pairs with distance > 4."""
        count = 0
        for src in range(n):
            dist = _bfs_distances(n, adj, src)
            for d in dist:
                if d > 4:
                    count += 1
        return count
    
    # Start with base graph
    base_adj = _build_adj(n, base_edges)
    
    # Greedy construction
    current_edges = []
    current_adj = [set(s) for s in base_adj]
    
    # Precompute far pairs for base
    def _get_far_pairs(adj):
        """Get set of unordered pairs with distance > 4."""
        far = set()
        for src in range(n):
            dist = _bfs_distances(n, adj, src)
            for dst in range(src+1, n):
                if dist[dst] > 4:
                    far.add((src, dst))
        return far
    
    far_pairs = _get_far_pairs(base_adj)
    
    # Greedy: repeatedly add the edge that covers the most far pairs
    while far_pairs and (time.time() - start_time) < deadline:
        best_edge = None
        best_cover = 0
        
        for edge in all_candidate_edges:
            if edge in [(min(u,v), max(u,v)) for u,v in current_edges]:
                continue
            # Check if adding this edge would create a triangle
            u, v = edge
            if current_adj[u] & current_adj[v]:
                continue
            
            # Estimate coverage: how many far pairs get fixed?
            # Approximate: pairs (a,b) where the new edge provides a shorter path
            # A pair (a,b) is covered if dist(a,u) + 1 + dist(v,b) <= 4 or dist(a,v) + 1 + dist(u,b) <= 4
            # Use current distances
            dist_from_u = _bfs_distances(n, current_adj, u)
            dist_from_v = _bfs_distances(n, current_adj, v)
            
            cover = 0
            for (a, b) in far_pairs:
                if dist_from_u[a] != -1 and dist_from_v[b] != -1:
                    if dist_from_u[a] + 1 + dist_from_v[b] <= 4:
                        cover += 1
                        break
                if dist_from_v[a] != -1 and dist_from_u[b] != -1:
                    if dist_from_v[a] + 1 + dist_from_u[b] <= 4:
                        cover += 1
                        break
            
            if cover > best_cover:
                best_cover = cover
                best_edge = edge
        
        if best_edge is None or best_cover == 0:
            break
        
        u, v = best_edge
        current_edges.append((u, v))
        current_adj[u].add(v)
        current_adj[v].add(u)
        
        # Update far pairs
        far_pairs = _get_far_pairs(current_adj)
    
    # Now we have a greedy solution. Try to reduce it.
    best = list(current_edges)
    
    # Try to remove edges
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            candidate = [e for j, e in enumerate(best) if j != i]
            if _validate_edges(candidate):
                best = candidate
                improved = True
                break
    
    # Try different greedy orderings / local search
    # Also try starting from different initial solutions
    
    # Try a few structured solutions
    structured_solutions = []
    
    # Solution 1: hub at 12
    sol1 = [(0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
            (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)]
    if _validate_edges(sol1):
        structured_solutions.append(list(sol1))
    
    # Solution 2: two hubs
    sol2 = [(0, 10), (2, 10), (4, 10), (14, 10), (16, 10), (18, 10), (20, 10), (22, 10),
            (23, 13), (21, 13), (19, 13), (17, 13), (15, 13)]
    if _validate_edges(sol2):
        structured_solutions.append(list(sol2))
    
    # Solution 3: ring-like
    sol3 = [(0, 8), (8, 16), (16, 23), (0, 12), (12, 23)]
    if _validate_edges(sol3):
        structured_solutions.append(list(sol3))
    
    # Solution 4: strategic long edges
    sol4 = [(0, 10), (10, 20), (20, 23), (0, 14), (14, 23)]
    if _validate_edges(sol4):
        structured_solutions.append(list(sol4))
    
    # Try to reduce each structured solution
    for sol in structured_solutions:
        if (time.time() - start_time) > deadline:
            break
        reduced = list(sol)
        imp = True
        while imp and (time.time() - start_time) < deadline:
            imp = False
            for i in range(len(reduced)):
                if (time.time() - start_time) > deadline:
                    break
                candidate = [e for j, e in enumerate(reduced) if j != i]
                if _validate_edges(candidate):
                    reduced = candidate
                    imp = True
                    break
        if len(reduced) < len(best):
            best = reduced
    
    # Local search: try swapping edges
    if (time.time() - start_time) < deadline:
        for _ in range(200):  # limited iterations
            if (time.time() - start_time) > deadline:
                break
            # Try replacing one edge with another
            for i in range(len(best)):
                if (time.time() - start_time) > deadline:
                    break
                for new_edge in all_candidate_edges:
                    if new_edge in [(min(u,v), max(u,v)) for u,v in best]:
                        continue
                    candidate = [e if j != i else new_edge for j, e in enumerate(best)]
                    if _validate_edges(candidate):
                        # Check if it's actually better or same size but we can continue
                        best = candidate
                        break
    
    # Final verification
    if _validate_edges(best):
        return [[u, v] for u, v in best]
    else:
        # Fallback
        fallback = [(0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
                    (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)]
        if _validate_edges(fallback):
            return [[u, v] for u, v in fallback]
        return []

# EVOLVE-BLOCK-END
