import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    deadline = time.time() + 80
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)

    def is_valid(added):
        # Check triangle free
        adj = [set(s) for s in base_adj]
        for u, v in added:
            if len(adj[u] & adj[v]) > 0:
                return False
            adj[u].add(v)
            adj[v].add(u)
            
        # Check diameter <= 4
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > 4:
                            return False
                        queue.append(v)
        return True

    best_solution = None

    # Search for 7-edge solution
    # We iterate over possible hubs 'c'
    for c in range(n):
        if time.time() > deadline:
            break
            
        # Candidates for edges from c: vertices v such that (c,v) is not in base and not triangle
        # Actually, we can just generate all possible sets of 7 edges from c to other vertices
        # and check validity.
        # But C(23, 7) is large.
        
        # Better: Use the domination logic.
        # For diameter 4 with hub c, we need to cover the path.
        # Vertices far from c need to be "pulled in" by other hubs or edges.
        # If we only use edges from c, then dist(u, c) must be small.
        # If dist(u, c) <= 2, then dist(u, v) <= 4 for all u, v.
        # We need to cover all 24 vertices with dist <= 2 from c using 7 edges.
        # Each edge (c, v) covers v-1, v, v+1.
        # We need to cover 0..23.
        # c is covered by itself (dist 0).
        # We need to cover [0, c-1] and [c+1, 23].
        
        # Let's just try all combinations of 7 vertices from 0..23 (excluding c)
        # and check if they form a valid star.
        # C(23, 7) = 245157. This is small enough to check quickly if is_valid is fast.
        
        candidates = [v for v in range(n) if v != c and (c, v) not in base_adj[c] and (v, c) not in base_adj[c]]
        # Note: base_adj[c] contains neighbors. We cannot add edge to neighbor.
        
        # Optimization: Only consider vertices that are not adjacent to c.
        # Also, to be triangle-free, the chosen vertices must not be adjacent to each other?
        # No, if we add (c, u) and (c, v), we need u and v to not be adjacent.
        # So we need a set of 7 vertices from candidates, no two adjacent.
        
        # Generate independent sets of size 7 from candidates?
        # Or just iterate combinations and check adjacency.
        
        for combo in combinations(candidates, 7):
            # Check mutual non-adjacency
            ok = True
            for i in range(7):
                for j in range(i+1, 7):
                    if combo[j] in base_adj[combo[i]]:
                        ok = False
                        break
                if not ok: break
            if not ok: continue
            
            edges = [(c, v) for v in combo]
            if is_valid(edges):
                best_solution = edges
                break
        if best_solution:
            break

    if best_solution is None:
        # Fallback: Try a known valid configuration or return empty (invalid)
        # If search failed, try a few specific hubs with manual tuning
        # For now, return a default 7-edge set that might work or empty
        # Since 7 is the min, and search should find it, this shouldn't happen.
        # But to be safe, return a hardcoded valid one if we can find one.
        # Let's try hub 11 with specific vertices again, but corrected.
        # The previous one failed on 0-9.
        # Try hub 12?
        # Or just return empty list if not found (score 0).
        # But we want score > 0.
        # I will trust the search.
        return []

    return [[int(u), int(v)] for u, v in best_solution]
