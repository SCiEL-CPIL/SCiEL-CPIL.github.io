"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    def is_valid(added_edges):
        # Check for duplicates and self-loops
        for u, v in added_edges:
            if u == v:
                return False
            if (u, v) in base_edges or (v, u) in base_edges:
                return False
            if (u, v) in added_edges or (v, u) in added_edges:
                return False
                
        # Build graph
        graph = {i: [] for i in range(n)}
        for u, v in base_edges:
            if v > u: # Avoid double adding
                graph[u].append(v)
                graph[v].append(u)
        for u, v in added_edges:
            graph[u].append(v)
            graph[v].append(u)
            
        # Check Connectivity
        visited = set([0])
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) != n:
            return False
            
        # Check Diameter
        max_dist = 0
        for start in range(n):
            dist = {start: 0}
            q = deque([start])
            while q:
                node = q.popleft()
                for neighbor in graph[node]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        max_dist = max(max_dist, dist[neighbor])
                        q.append(neighbor)
            if max_dist > 4:
                return False
                
        # Check Triangle Free (Bipartite check is faster, but let's just check triangles for safety if N is small)
        # Since we only add cross-parity edges, it should be bipartite.
        # Let's verify bipartiteness.
        color = {}
        for i in range(n):
            if i not in color:
                color[i] = 0
                q = deque([i])
                while q:
                    node = q.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[node]
                            q.append(neighbor)
                        else:
                            if color[neighbor] == color[node]:
                                return False
        return True

    best_solution = []
    best_cost = float('inf')
    
    # Strategy: Central Clusters
    # Hubs must be close to each other.
    # Try different sets of hubs in the center.
    
    center_indices = [10, 11, 12, 13]
    
    # Generate candidate hub sets
    # H_E subset of Evens, H_O subset of Odds
    # They must be "connected" in the hub graph.
    
    candidates = []
    
    # Try single hub pairs
    for h_e in [10, 12, 14]:
        for h_o in [9, 11, 13, 15]:
            if abs(h_e - h_o) <= 2: # Ensure hub distance is small
                candidates.append((frozenset([h_e]), frozenset([h_o])))
                
    # Try small clusters
    # H_E size 2, H_O size 2
    for h_e1 in [10, 12]:
        for h_e2 in [12, 14]:
            if h_e1 == h_e2: continue
            for h_o1 in [9, 11, 13]:
                for h_o2 in [11, 13, 15]:
                    if h_o1 == h_o2: continue
                    # Check if all pairs are dist <= 2
                    ok = True
                    for he in [h_e1, h_e2]:
                        for ho in [h_o1, h_o2]:
                            if abs(he - ho) > 2: # Rough check, actual dist in path is abs diff
                                # Wait, dist in path is abs diff.
                                # But we might have added edges between hubs?
                                # No, we don't add edges between hubs in this model, we rely on path.
                                # If abs diff > 2, dist > 2.
                                ok = False
                    if ok:
                        candidates.append((frozenset([h_e1, h_e2]), frozenset([h_o1, h_o2])))
                        
    for h_e_set, h_o_set in candidates:
        added = []
        # Add edges for E not covered
        for v in range(0, n, 2):
            if v in h_e_set:
                continue
            # Check if v is dist 1 from any h in h_o_set
            covered = False
            for h in h_o_set:
                if (v, h) in base_edges or (h, v) in base_edges:
                    covered = True
                    break
            if not covered:
                # Add edge to closest hub in h_o_set?
                # Or just pick one.
                # To minimize edges, we just add ONE edge per uncovered v.
                # Pick the first one.
                added.append([v, list(h_o_set)[0]])
                
        # Add edges for O not covered
        for v in range(1, n, 2):
            if v in h_o_set:
                continue
            covered = False
            for h in h_e_set:
                if (v, h) in base_edges or (h, v) in base_edges:
                    covered = True
                    break
            if not covered:
                added.append([v, list(h_e_set)[0]])
                
        # Check validity
        if is_valid(added):
            if len(added) < best_cost:
                best_cost = len(added)
                best_solution = added
                
    if not best_solution:
        # Fallback: The 20-edge solution (11 to all E, 12 to all O)
        added = []
        for v in range(0, n, 2):
            if (11, v) not in base_edges and (v, 11) not in base_edges:
                added.append([11, v])
        for v in range(1, n, 2):
            if (12, v) not in base_edges and (v, 12) not in base_edges:
                added.append([12, v])
        best_solution = added
        
    return best_solution
# EVOLVE-BLOCK-END
