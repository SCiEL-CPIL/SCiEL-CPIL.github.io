import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = [(e[0], e[1]) for e in P['edges']]
    base_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in base_edges)
    
    # Strategy: Hub-and-spoke from center
    # The path is bipartite. Let's pick a central vertex, e.g., 12.
    # Connect 12 to all vertices of opposite parity that are not adjacent to it.
    # Adjacent to 12 are 11 and 13.
    # Parity of 12 is even. Opposite parity is odd.
    # Odd vertices: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23.
    # Exclude 11, 13.
    # Connect 12 to: 1, 3, 5, 7, 9, 15, 17, 19, 21, 23.
    # This creates a star-like structure around 12 for the odd vertices.
    # Distance from 12 to any odd vertex is 1.
    # Distance from any even vertex to 12 is at most 2 (via path or via odd neighbor).
    # Actually, let's verify diameter.
    # Max distance is between two odd vertices far apart, e.g., 1 and 23.
    # 1 -> 12 (1) -> 23 (1). Total 2.
    # Max distance between even vertices, e.g., 0 and 22.
    # 0 -> 1 -> 12 -> 21 -> 22? No, 21 is odd.
    # 0 -> 1 (1) -> 12 (1) -> 21 (1) -> 22 (1)?
    # Wait, 12 is connected to 21.
    # 0-1-12-21-22. Length 4.
    # 0-1-2-3...
    # Let's check 0 and 22.
    # 0 is even. 22 is even.
    # Path: 0-1-12-21-22. Length 4.
    # Is there a shorter one?
    # 0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22.
    # With shortcuts:
    # 0-1-12-21-22 is length 4.
    # What about 0 and 23?
    # 0-1-12-23. Length 3.
    # What about 2 and 22?
    # 2-3-12-21-22. Length 4.
    # What about 0 and 20?
    # 0-1-12-19-20. Length 4.
    # It seems diameter is 4.
    
    # The added edges are:
    # (12, 1), (12, 3), (12, 5), (12, 7), (12, 9), (12, 15), (12, 17), (12, 19), (12, 21), (12, 23)
    
    added_edges = []
    center = 12
    for v in range(n):
        if v == center:
            continue
        if v % 2 != center % 2: # Opposite parity
            # Check if adjacent in base path
            if abs(v - center) != 1:
                # Check if already in base_set (shouldn't be for non-adjacent)
                key = (min(center, v), max(center, v))
                if key not in base_set:
                    added_edges.append([int(center), int(v)])
                    
    # Verify constraints just in case
    # 1. No duplicates with base
    # 2. Triangle free
    # 3. Diameter <= 4
    
    # Since the construction is deterministic and proven logic, we can return directly.
    # But for safety, let's do a quick validation.
    
    def _validate(n, base_edges, added_edges):
        all_edges = list(base_edges) + list(added_edges)
        seen = set()
        for u, v in all_edges:
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
            
        # Triangle free check
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
                        
        # Diameter check
        from collections import deque
        max_dist = 0
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
            if -1 in dist:
                return False
            max_dist = max(max_dist, max(dist))
            
        return max_dist <= 4

    if _validate(n, base_edges, added_edges):
        return added_edges
    
    # Fallback if validation fails (should not happen)
    return []

# EVOLVE-BLOCK-END
