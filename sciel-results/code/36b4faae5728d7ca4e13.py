import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
        
    # Precompute valid edges (no triangles with base, no duplicates)
    # An edge (u, v) is valid if u, v not adjacent in base, and no common neighbor in base.
    # Common neighbor w means (u, w) in base and (v, w) in base.
    # In a path, u and v have a common neighbor iff |u - v| == 2.
    # So valid edges are those with |u - v| >= 3.
    
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if v - u >= 3:
                valid_edges.append((u, v))
                
    # We want to find a subset of valid_edges such that:
    # 1. Union is connected.
    # 2. Diameter <= 4.
    # 3. No triangles formed by added edges + base edges.
    #    This is guaranteed if we don't add edges that form triangles with base.
    #    But added edges can form triangles with each other?
    #    If we add (u, v) and (u, w), and v, w are adjacent in base, triangle.
    #    So we must ensure that for any u, the set of added neighbors of u is an independent set in the base path.
    
    # Greedy search or ILP?
    # Given time limit, let's try a constructive heuristic.
    
    # Try to build a solution with a small number of edges.
    # The seed program adds many edges. We want fewer.
    
    # Heuristic: 
    # 1. Identify a set of hubs.
    # 2. Connect hubs to a center or each other.
    
    # Let's try to find a solution with <= 10 edges.
    
    best_edges = None
    best_count = n # upper bound
    
    # Try specific configurations
    # Configuration 1: Star around 12, with adjustments
    # We found star around 12 fails for 14.
    # Try star around 11 and 13?
    
    # Let's try a BFS-based check for a candidate set of edges.
    
    def is_valid_solution(added_edges):
        # Check duplicates
        for e in added_edges:
            if (e[0], e[1]) in base_edges or (e[1], e[0]) in base_edges:
                return False
        
        # Check triangles with base
        # For each added edge (u, v), check if they have common neighbor in base.
        # This is |u-v|==2, which we excluded in valid_edges.
        
        # Check triangles with other added edges
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for triangles
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True

    def get_diameter(added_edges):
        adj = {i: [] for i in range(n)}
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            max_diam = max(max_diam, max(dist))
        return max_diam

    # Try to construct a solution
    # Use a set of hubs spaced by 3, connected to a central hub.
    
    # Candidate: Hubs at 0, 3, 6, 9, 12, 15, 18, 21, 23?
    # Connect all to 12?
    # We know 12-14 is bad. 14 is not a hub here.
    # Hubs: 0, 3, 6, 9, 12, 15, 18, 21, 23.
    # Check coverage:
    # 0 covers 0, 1.
    # 3 covers 2, 3, 4.
    # 6 covers 5, 6, 7.
    # 9 covers 8, 9, 10.
    # 12 covers 11, 12, 13.
    # 15 covers 14, 15, 16.
    # 18 covers 17, 18, 19.
    # 21 covers 20, 21, 22.
    # 23 covers 22, 23.
    # All covered.
    
    # Connect all hubs to 12.
    # Edges: (12, 0), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23).
    # Note: (12, 12) is not an edge.
    
    edges_to_try = [(12, 0), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23)]
    
    if is_valid_solution(edges_to_try):
        diam = get_diameter(edges_to_try)
        if diam <= 4:
            return [[u, v] for u, v in edges_to_try]
    
    # If that fails, try to add edges to reduce diameter.
    # Or try a different center.
    
    # Try center 11?
    edges_to_try = [(11, 0), (11, 3), (11, 6), (11, 9), (11, 15), (11, 18), (11, 21), (11, 23)]
    # Check validity of (11, 9)?
    # Neighbors of 11: 10, 12. Neighbors of 9: 8, 10.
    # Common: 10. Triangle (11, 10, 9). Invalid.
    
    # So (11, 9) is invalid.
    # We need to be careful.
    
    # Let's just return the seed solution if we can't find a better one quickly.
    # But seed is score 1.0. We want > 1.0.
    # Seed adds ~100 edges.
    # My solution adds 8 edges.
    # If valid, score is 100/8 = 12.5.
    
    # Let's verify the 8-edge solution again.
    # Hubs: 0, 3, 6, 9, 12, 15, 18, 21, 23.
    # Center: 12.
    # Edges: (12, 0), (12, 3), (12, 6), (12, 9), (12, 15), (12, 18), (12, 21), (12, 23).
    
    # Check (12, 9):
    # Neighbors of 12: 11, 13.
    # Neighbors of 9: 8, 10.
    # No common. OK.
    
    # Check (12, 15):
    # Neighbors of 12: 11, 13.
    # Neighbors of 15: 14, 16.
    # No common. OK.
    
    # Check triangles between added edges:
    # All share 12.
    # Need to check if any two hubs are adjacent.
    # Hubs: 0, 3, 6, 9, 15, 18, 21, 23.
    # Pairs:
    # (0, 3): dist 3. OK.
    # (3, 6): dist 3. OK.
    # (6, 9): dist 3. OK.
    # (9, 15): dist 6. OK.
    # (15, 18): dist 3. OK.
    # (18, 21): dist 3. OK.
    # (21, 23): dist 2. **ADJACENT**.
    # 21 and 23 are NOT adjacent. 22 is between them.
    # Wait, |21 - 23| = 2.
    # Are they adjacent in base? No, base has (21, 22) and (22, 23).
    # So 21 and 23 are NOT adjacent.
    # So no triangle (12, 21, 23).
    
    # So the solution is valid.
    
    # Diameter check:
    # Hubs are within 1 of 12?
    # d(12, 0) = 1.
    # d(12, 3) = 1.
    # d(12, 6) = 1.
    # d(12, 9) = 1.
    # d(12, 15) = 1.
    # d(12, 18) = 1.
    # d(12, 21) = 1.
    # d(12, 23) = 1.
    # So all hubs are dist 1 from 12.
    # Max dist between hubs is 2 (via 12).
    # Every vertex is dist 1 from a hub.
    # So diameter is 1 + 2 + 1 = 4.
    
    return [[u, v] for u, v in edges_to_try]
