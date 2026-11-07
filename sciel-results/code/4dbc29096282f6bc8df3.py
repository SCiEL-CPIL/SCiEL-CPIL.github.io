"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj(n, base_edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _bfs_diameter(n, adj):
    max_diam = 0
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
        for d in dist:
            if d == -1:
                return -1
            if d > max_diam:
                max_diam = d
    return max_diam

def _is_valid(n, added_edges, base_adj):
    adj = [row[:] for row in base_adj]
    for u, v in added_edges:
        if not isinstance(u, int) or not isinstance(v, int):
            return False
        if u == v:
            return False
        if adj[u][v]:
            return False
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                return False
        adj[u][v] = True
        adj[v][u] = True
    diam = _bfs_diameter(n, adj)
    return diam <= 4

def _can_add(adj, u, v, n):
    if adj[u][v]:
        return False
    for w in range(n):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    base_adj = _build_base_adj(n, P['edges'])
    
    # Precompute which edges are valid to add (no triangle with base graph)
    valid_single = []
    for u in range(n):
        for v in range(u + 1, n):
            if not base_adj[u][v] and _can_add(base_adj, u, v, n):
                valid_single.append((u, v))
    
    # Start with known valid solution of 8 edges
    seed = [(12, 1), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    best = list(seed)
    best_count = len(best)
    
    # Try to find solutions with fewer edges using systematic search
    # Strategy: for each possible hub vertex, try to find minimal set of edges
    # that connects all vertices within distance 4
    
    # First, let's try to reduce from 8 to 7
    for i in range(len(best)):
        if time.time() > deadline:
            break
        current = [e for j, e in enumerate(best) if j != i]
        current_set = set(current)
        
        found = False
        for cand in valid_single:
            if time.time() > deadline:
                break
            if cand in current_set:
                continue
            new_edges = current + [cand]
            if _is_valid(n, new_edges, base_adj):
                best = new_edges
                best_count = len(best)
                found = True
                break
        
        if found:
            break
    
    # Try to reduce further: 7 to 6
    if best_count == 7:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            current = [e for j, e in enumerate(best) if j != i]
            current_set = set(current)
            
            found = False
            for cand in valid_single:
                if time.time() > deadline:
                    break
                if cand in current_set:
                    continue
                new_edges = current + [cand]
                if _is_valid(n, new_edges, base_adj):
                    best = new_edges
                    best_count = len(best)
                    found = True
                    break
            
            if found:
                break
    
    # Try to reduce further: 6 to 5
    if best_count == 6:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            current = [e for j, e in enumerate(best) if j != i]
            current_set = set(current)
            
            found = False
            for cand in valid_single:
                if time.time() > deadline:
                    break
                if cand in current_set:
                    continue
                new_edges = current + [cand]
                if _is_valid(n, new_edges, base_adj):
                    best = new_edges
                    best_count = len(best)
                    found = True
                    break
            
            if found:
                break
    
    # Try a different approach: systematic search for smaller solutions
    # Try all hub vertices and find minimal edge sets
    if time.time() < deadline - 10:
        for hub in range(n):
            if time.time() > deadline - 5:
                break
            
            # Find all vertices that can be connected to hub without creating triangles
            hub_edges = []
            for v in range(n):
                if v != hub and not base_adj[hub][v] and _can_add(base_adj, hub, v, n):
                    hub_edges.append(v)
            
            # Try to find minimal subset of hub_edges that gives diameter <= 4
            # Use greedy: sort by distance from hub in base graph, pick farthest first
            # But we need to be smart about which vertices to connect
            
            # Actually, let's try a different strategy:
            # For a given hub, the vertices not directly connected to hub must be
            # within distance 3 of hub via the path + other added edges
            
            # Let's try all subsets of size 5, 6, 7 from hub_edges
            # But this could be expensive. Let's be smarter.
            
            # Key insight: if we connect hub to a set S of vertices, then any vertex v
            # not in S must be within distance 3 of hub in the augmented graph.
            # In the base path, dist_path(hub, v) = |hub - v|.
            # With added edges, the distance can be shorter.
            
            # For a vertex v, dist_aug(hub, v) <= 4 means:
            # - v is in S (dist 1), or
            # - there exists u in S such that dist_path(v, u) <= 3 (dist 2 via u), or
            # - there exists u in S and w such that dist_path(v, u) + dist_path(u, w) <= 3... 
            # This is getting complex. Let's just try small subsets.
            
            # Try subsets of size 5 first
            if len(hub_edges) >= 5:
                # Use a heuristic: pick vertices that cover the most distance
                # Sort hub_edges by |hub - v| descending
                sorted_he = sorted(hub_edges, key=lambda x: abs(hub - x), reverse=True)
                
                # Try the top 5
                candidate_set = sorted_he[:5]
                test_edges = [(hub, v) for v in candidate_set]
                if _is_valid(n, test_edges, base_adj):
                    if len(test_edges) < best_count:
                        best = test_edges
                        best_count = len(best)
            
            # Try subsets of size 6
            if len(hub_edges) >= 6 and best_count > 6:
                sorted_he = sorted(hub_edges, key=lambda x: abs(hub - x), reverse=True)
                candidate_set = sorted_he[:6]
                test_edges = [(hub, v) for v in candidate_set]
                if _is_valid(n, test_edges, base_adj):
                    if len(test_edges) < best_count:
                        best = test_edges
                        best_count = len(best)
            
            # Try subsets of size 7
            if len(hub_edges) >= 7 and best_count > 7:
                sorted_he = sorted(hub_edges, key=lambda x: abs(hub - x), reverse=True)
                candidate_set = sorted_he[:7]
                test_edges = [(hub, v) for v in candidate_set]
                if _is_valid(n, test_edges, base_adj):
                    if len(test_edges) < best_count:
                        best = test_edges
                        best_count = len(best)
    
    # Final verification
    if not _is_valid(n, best, base_adj):
        # Fallback to seed
        best = list(seed)
    
    return [list(e) for e in best]

# EVOLVE-BLOCK-END
