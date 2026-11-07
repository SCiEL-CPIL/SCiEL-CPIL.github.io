"""Rewritten program for triangle-free diameter-four augmentation."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # Build adjacency for base path
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # A candidate edge (u,v) is valid if it doesn't create a triangle
    # i.e., u and v share no common neighbor in the base graph (since we start from base)
    # But actually, as we add edges, the triangle-free constraint is on the FULL graph.
    # For the path, u and v are adjacent in the path only if |u-v|==1.
    # A triangle (a,b,c) exists if all three pairs are edges.
    # For a path, adding edge (u,v) creates a triangle if there exists w such that
    # (u,w) and (v,w) are both in the current graph.
    
    # Let's think about this differently. The base is a path. 
    # The only way to create a triangle with a single new edge (u,v) in a path is
    # if there's a vertex w adjacent to both u and v in the path, i.e., w is a common neighbor.
    # In a path, each vertex has at most 2 neighbors. So (u,v) creates a triangle iff
    # dist_path(u,v) == 2 (they share a middle vertex).
    # Wait, if we add multiple edges, new triangles can form involving multiple new edges.
    # E.g., add (0,2), (2,4), (0,4) -> triangle (0,2,4).
    
    # So the triangle-free constraint is on the final graph.
    
    # First, find which pairs need to be covered: pairs with path distance > 4.
    # In a path 0-1-...-23, dist(i,j) = |i-j|.
    # So we need all pairs (i,j) with |i-j| > 4 to have distance <= 4 in the augmented graph.
    
    # Let's use a greedy + local search approach.
    # Start with no extra edges, then greedily add edges that reduce the number of "far" pairs the most.
    
    def bfs_diameter(adj):
        """Compute diameter via BFS from each node."""
        n = len(adj)
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        queue.append(nei)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    def is_triangle_free(adj):
        """Check if graph is triangle-free."""
        n = len(adj)
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                if common:
                    return False
        return True
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def count_far_pairs(adj, threshold=4):
        """Count pairs with distance > threshold."""
        n = len(adj)
        far_count = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = [i]
            for node in queue:
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        queue.append(nei)
            for j in range(i+1, n):
                if dist[j] > threshold:
                    far_count += 1
        return far_count
    
    # Candidate edges: all non-adjacent pairs in the path that don't create a triangle with base
    # Actually, any pair (u,v) with |u-v| >= 2 is a candidate (|u-v|==1 are base edges)
    # But (u,v) with |u-v|==2 shares a neighbor with base, so adding it alone doesn't create triangle
    # in base, but it's still a valid candidate edge. The triangle-free check is on the full graph.
    
    # Let's generate all candidate edges
    candidates = []
    for u in range(n):
        for v in range(u+2, n):  # skip adjacent (base edges)
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))
    
    # Sort candidates by "potential" - longer edges are more useful
    # Actually, let's use a smarter ordering: prioritize edges that cover many far pairs
    
    deadline = time.time() + 45  # leave margin for safety
    
    # Greedy: start with empty set, add edges one by one
    current_edges = []
    current_adj = build_adj(current_edges)
    # Merge with base
    for u, v in base_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
    
    current_far = count_far_pairs(current_adj)
    
    while time.time() < deadline:
        # Find the edge that reduces far pairs the most
        best_edge = None
        best_reduction = 0
        best_far_after = current_far
        
        for edge in candidates:
            if edge in current_edges:
                continue
            u, v = edge
            # Temporarily add and check
            trial_adj = [set(s) for s in current_adj]
            trial_adj[u].add(v)
            trial_adj[v].add(u)
            
            if not is_triangle_free(trial_adj):
                continue
            
            trial_far = count_far_pairs(trial_adj)
            reduction = current_far - trial_far
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_edge = edge
                best_far_after = trial_far
        
        if best_edge is None or best_reduction == 0:
            break
        
        current_edges.append(best_edge)
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_far = best_far_after
    
    # Now try to remove unnecessary edges
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for i in range(len(current_edges) - 1, -1, -1):
            trial_edges = current_edges[:i] + current_edges[i+1:]
            trial_adj = build_adj(trial_edges)
            for u, v in base_edges:
                trial_adj[u].add(v)
                trial_adj[v].add(u)
            if is_triangle_free(trial_adj) and bfs_diameter(trial_adj) <= 4:
                current_edges = trial_edges
                removed = True
                break
    
    # Local search: try swapping edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_edges)):
            for j in range(len(candidates)):
                if candidates[j] in current_edges:
                    continue
                trial_edges = current_edges[:i] + current_edges[i+1:] + [candidates[j]]
                trial_adj = build_adj(trial_edges)
                for u, v in base_edges:
                    trial_adj[u].add(v)
                    trial_adj[v].add(u)
                if is_triangle_free(trial_adj) and bfs_diameter(trial_adj) <= 4:
                    # Try to remove more
                    trial_edges2 = list(trial_edges)
                    for k in range(len(trial_edges2) - 1, -1, -1):
                        trial2 = trial_edges2[:k] + trial_edges2[k+1:]
                        trial_adj2 = build_adj(trial2)
                        for u, v in base_edges:
                            trial_adj2[u].add(v)
                            trial_adj2[v].add(u)
                        if is_triangle_free(trial_adj2) and bfs_diameter(trial_adj2) <= 4:
                            trial_edges2 = trial2
                    if len(trial_edges2) < len(current_edges):
                        current_edges = trial_edges2
                        improved = True
                        break
            if improved:
                break
    
    # Final validation
    final_adj = build_adj(current_edges)
    for u, v in base_edges:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    assert is_triangle_free(final_adj), "Triangle found"
    assert bfs_diameter(final_adj) <= 4, f"Diameter too large: {bfs_diameter(final_adj)}"
    
    # Convert to proper format
    result = []
    for u, v in current_edges:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
