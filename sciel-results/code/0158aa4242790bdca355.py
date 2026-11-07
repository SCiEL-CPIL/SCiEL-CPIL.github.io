"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(frozenset(e))
    
    # Build adjacency for base graph
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    
    # Generate candidate edges (all non-base edges)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if frozenset((u,v)) not in base_edges:
                candidates.append((u,v))
    
    # For each candidate, check if adding it creates a triangle
    # A triangle is created if there exists w such that w is adjacent to both u and v
    valid_candidates = []
    for u, v in candidates:
        # Check if u and v have a common neighbor in the base graph
        # If they do, adding (u,v) creates a triangle
        common = adj[u] & adj[v]
        if not common:
            valid_candidates.append((u,v))
    
    # Now we need to select minimum subset of valid_candidates such that:
    # 1. The resulting graph has diameter <= 4
    # 2. No triangles are created (but we need to check dynamically since adding one edge might affect triangle-freeness of another)
    
    # Actually, triangle-freeness is not just about individual edges. Adding edge (u,v) is safe only if no common neighbor exists in the CURRENT graph.
    # So we need to be more careful.
    
    # Let's use a greedy approach: try to find a small set that works.
    
    # First, let's think about what structure we need.
    # Diameter 4 means every pair of vertices is within distance 4.
    # In the path, the farthest pair is (0,23) with distance 23.
    
    # Strategy: Add edges to create shortcuts.
    # Let's try to find a set of edges that makes the diameter <= 4.
    
    # We'll use a BFS-based check for diameter.
    
    def check_diameter(edge_set):
        """Check if graph with base edges + edge_set has diameter <= 4."""
        adj_check = [set() for _ in range(n)]
        for e in P['edges']:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        for e in edge_set:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adj_check[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > 4:
                            return False
                        queue.append(nb)
            if any(d == -1 for d in dist):
                return False
        return True
    
    def check_triangle_free(edge_set):
        """Check if graph with base edges + edge_set is triangle-free."""
        adj_check = [set() for _ in range(n)]
        for e in P['edges']:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        for e in edge_set:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        
        for u in range(n):
            for v in adj_check[u]:
                if v <= u:
                    continue
                for w in adj_check[u]:
                    if w <= u:
                        continue
                    if w in adj_check[v]:
                        return False
        return True
    
    # Greedy approach: start with no edges, add edges that help most
    # Actually, let's try a smarter approach.
    
    # Let's try specific patterns.
    # Idea: connect vertices that are far apart in the path.
    
    # Let me try: add edges (0,4), (0,8), (4,8) -- no wait, that creates triangle 0-4-8-0
    # So we can't add all three.
    
    # Let's think: in a triangle-free graph, if we have a vertex connected to many others, those others can't be connected to each other.
    
    # Try: make vertex 12 a hub. Connect 12 to 0, 4, 8, 16, 20, 23.
    # Check triangles: 
    # 12-0: 0's neighbors: 1. 12's neighbors: 11,13. No common. OK.
    # 12-4: 4's neighbors: 3,5. 12's neighbors: 11,13,0 (if 12-0 added). 0 not neighbor of 4. OK.
    # 12-8: 8's neighbors: 7,9. 12's neighbors: 11,13,0,4. None common. OK.
    # 12-16: 16's neighbors: 15,17. 12's neighbors: 11,13,0,4,8. None common. OK.
    # 12-20: 20's neighbors: 19,21. 12's neighbors: 11,13,0,4,8,16. None common. OK.
    # 12-23: 23's neighbors: 22. 12's neighbors: 11,13,0,4,8,16,20. None common. OK.
    
    # Now check diameter with these 6 edges.
    test_edges = [(0,12), (4,12), (8,12), (12,16), (12,20), (12,23)]
    if check_diameter(test_edges) and check_triangle_free(test_edges):
        return [[u,v] for u,v in test_edges]
    
    # If not, try other combinations.
    # Let's try a more systematic search.
    
    # Try different hub configurations.
    
    # Let's try: two hubs.
    # Hub 1: vertex 6, connected to 0, 2, 4, 10, 14, 18
    # Hub 2: vertex 18, connected to 6, 10, 14, 20, 22, 23
    # Check triangles...
    
    # This is getting complex. Let me just do a more thorough search.
    
    # Try all subsets of size k for small k.
    
    # But 24 vertices, ~276 candidate edges, so we can't do exhaustive.
    
    # Let's try a greedy approach:
    # Start with empty set. At each step, add the edge that most reduces the diameter.
    
    def get_diameter(edge_set):
        adj_check = [set() for _ in range(n)]
        for e in P['edges']:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        for e in edge_set:
            u, v = e
            adj_check[u].add(v)
            adj_check[v].add(u)
        
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adj_check[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        max_d = max(max_d, dist[nb])
                        queue.append(nb)
        return max_d
    
    current_edges = set()
    current_diam = get_diameter(current_edges)
    
    while current_diam > 4:
        best_edge = None
        best_diam = current_diam + 1
        for u, v in candidates:
            if (u,v) in current_edges or (v,u) in current_edges:
                continue
            new_edges = current_edges | {(u,v)}
            # Check triangle-free
            if not check_triangle_free(new_edges):
                continue
            d = get_diameter(new_edges)
            if d < best_diam:
                best_diam = d
                best_edge = (u,v)
        if best_edge is None:
            break
        current_edges.add(best_edge)
        current_diam = best_diam
    
    if current_diam <= 4 and check_triangle_free(current_edges):
        return [[u,v] for u,v in current_edges]
    
    # Fallback: return the original approach
    return [[u,v] for u in range(n) for v in range(u+1,n) if (u-v)%2 and (u,v) not in {tuple(e) for e in P['edges']}]
# EVOLVE-BLOCK-END
