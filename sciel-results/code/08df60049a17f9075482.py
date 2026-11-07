import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    start_time = time.time()
    deadline = start_time + 85

    # Bipartite: even vertices and odd vertices
    # Only add edges between even and odd vertices to guarantee triangle-free
    even_verts = [i for i in range(n) if i % 2 == 0]
    odd_verts = [i for i in range(n) if i % 2 == 1]
    
    # All possible bipartite edges not in base
    bipartite_candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                bipartite_candidates.append((min(u,v), max(u,v)))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        queue.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist

    def is_valid(added_edges):
        """Check if adding these bipartite edges gives connected graph with diameter <= 4."""
        adj = build_adj(added_edges)
        # Check connected
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        # Check diameter
        return get_diameter(adj) <= 4

    def greedy_build():
        """Greedy: repeatedly add the edge that most reduces the number of pairs with dist > 4."""
        added = set()
        adj = build_adj(added)
        
        while True:
            if time.time() > deadline:
                break
            
            # Compute all-pairs distances
            all_dists = []
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
                all_dists.append(dist)
            
            # Count far pairs
            far_count = sum(1 for i in range(n) for j in range(i+1, n) if all_dists[i][j] > 4)
            if far_count == 0:
                break
            
            # Find best edge to add
            best_edge = None
            best_far_count = far_count
            
            for u, v in bipartite_candidates:
                if (u, v) in added:
                    continue
                # Check if this edge is still available (not in base)
                if (u, v) in base_edges:
                    continue
                # Temporarily add and count far pairs
                new_adj = [set(adj[i]) for i in range(n)]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                new_far = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    q = deque([src])
                    while q:
                        node = q.popleft()
                        for nb in new_adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                if dist[nb] > 4:
                                    pass  # still count it
                                q.append(nb)
                    for j in range(src+1, n):
                        if dist[j] > 4:
                            new_far += 1
                
                if new_far < best_far_count:
                    best_far_count = new_far
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            added.add(best_edge)
            adj = build_adj(added)
        
        return added

    def local_search_reduce(added_edges):
        """Try to remove edges while maintaining validity."""
        added = set(added_edges)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added):
                if time.time() > deadline:
                    break
                new_set = added - {edge}
                if is_valid(new_set):
                    added = new_set
                    improved = True
                    break
        return added

    # Known good solution as fallback
    hub_solution = [(1, 11), (4, 11), (7, 11), (11, 15), (11, 18), (11, 21), (11, 23)]
    best = set(hub_solution)
    best_count = len(best)
    
    # Try greedy build
    greedy_result = greedy_build()
    if greedy_result:
        greedy_result = local_search_reduce(greedy_result)
        if is_valid(greedy_result) and len(greedy_result) < best_count:
            best = greedy_result
            best_count = len(best)
    
    # Try systematic search for small k
    # For k=2,3,4: try all combinations of bipartite candidates
    # But there are ~144 bipartite candidates, so C(144,4) is too large
    # Use smarter approach: try hub-based structures
    
    # Try different hub configurations
    # A hub at position h connected to vertices within distance 4 on both sides
    # For diameter 4, we might need 2 hubs
    
    # Try single hub approaches
    for h in range(n):
        if time.time() > deadline:
            break
        # Connect hub to vertices that need it
        added = set()
        for v in range(n):
            if v == h:
                continue
            # Distance from h to v in path is |h-v|
            # With edge (h,v), dist becomes 1
            # We need all pairs within 4
            # For a pair (a,b), dist = min(|a-b|, min(dist(a,h)+1+dist(h,b) via edge...))
            # This is complex, so just try connecting h to all vertices at distance > 2
            if abs(h - v) > 2 and (min(h,v), max(h,v)) not in base_edges:
                # Check bipartite
                if (h % 2) != (v % 2):
                    added.add((min(h,v), max(h,v)))
        
        if added and is_valid(added):
            added = local_search_reduce(added)
            if is_valid(added) and len(added) < best_count:
                best = added
                best_count = len(best)
    
    # Try two-hub approach
    for h1 in range(n):
        if time.time() > deadline:
            break
        for h2 in range(h1+1, n):
            if time.time() > deadline:
                break
            added = set()
            for v in range(n):
                if v == h1 or v == h2:
                    continue
                # Connect to nearest hub if distance > 2
                d1 = abs(h1 - v)
                d2 = abs(h2 - v)
                if d1 <= d2 and d1 > 2 and (h1 % 2) != (v % 2):
                    added.add((min(h1,v), max(h1,v)))
                elif d2 < d1 and d2 > 2 and (h2 % 2) != (v % 2):
                    added.add((min(h2,v), max(h2,v)))
            
            if added and is_valid(added):
                added = local_search_reduce(added)
                if is_valid(added) and len(added) < best_count:
                    best = added
                    best_count = len(best)
    
    # Final local search on best
    best = local_search_reduce(best)
    
    # Convert to list of [int, int]
    result = []
    for u, v in best:
        result.append([int(u), int(v)])
    
    return result
