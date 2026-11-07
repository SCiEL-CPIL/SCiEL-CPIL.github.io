import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = 85.0

    n = P['n']
    base_edges = P['edges']

    # Initialize adjacency from base path
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def bfs_dist(adj, src):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def check_diameter(adj, limit=4):
        for s in range(n):
            d = bfs_dist(adj, s)
            if -1 in d:
                return False
            if max(d) > limit:
                return False
        return True

    def is_triangle_free(adj, added):
        # Check only triangles involving at least one added edge
        for u, v in added:
            common = adj[u] & adj[v]
            if len(common) > 0:
                return False
        return True

    # Strategy: Multi-hub initialization with greedy removal
    # The path is bipartite with even/odd parts.
    # Adding edges between same-parity vertices preserves bipartiteness (triangle-free).
    # We want to minimize edges while keeping diameter <= 4.
    #
    # Key insight: For a path of length 23 (24 vertices), the worst-case distance is 23.
    # We need to reduce all pairwise distances to <= 4.
    #
    # Approach:
    # 1. Start with a set of "long chords" that connect vertices far apart on the path.
    # 2. Use greedy removal to eliminate unnecessary edges.
    #
    # Let's try a structured approach:
    # Add edges (i, i+k) for various k to create shortcuts.
    # Then remove edges that aren't needed.

    # Candidate edges to consider: all non-adjacent pairs on the path
    # But we need a smart initial set.
    #
    # For diameter 4 on a path of 24 vertices:
    # We can think of placing "hubs" at regular intervals.
    # If we add edges (0,12), (1,13), (2,14), etc., that might help but uses many edges.
    #
    # Better: Use a small number of long chords.
    # Let's try: connect vertices at distance ~8-10 apart.
    #
    # Actually, let's use a different approach:
    # Start with NO added edges, then greedily ADD edges to reduce diameter.
    # At each step, pick the edge that most reduces the "excess diameter" (sum of distances above 4, or max distance).
    #
    # But greedy add might be slow. Let's try greedy REMOVE from a known-good set.
    
    # Known-good set: connect every vertex to a central hub, but only if it helps.
    # Actually, let's try a sparse structured set first.
    
    # Try: add edges (i, i+8) for i in 0..15, and (i, i+7) for some
    # That's 16+7=23 edges. Too many.
    
    # Let's try: add edges that create a "ladder" pattern.
    # (0,8), (1,9), (2,10), ..., (15,23) - that's 16 edges
    # (0,12), (1,13), (2,14), ..., (11,23) - that's 12 edges
    # Total ~28 edges. Still too many.
    
    # Let me think differently. What's the minimum number of edges needed?
    # With k added edges, we have 23+k total edges.
    # The diameter constraint is very tight.
    
    # Let's use a constructive approach:
    # Place "jump" edges at strategic positions.
    
    # Start with a reasonable initial set:
    # Connect vertices that are 8 apart: (0,8), (1,9), ..., (15,23)
    # This gives us a "2-lane" path with diameter roughly ceil(23/2) = 12. Still too high.
    
    # Connect vertices that are 12 apart: (0,12), (1,13), ..., (11,23)
    # This gives diameter roughly ceil(23/2) still.
    
    # We need both: some 8-jumps and some 12-jumps, or a different pattern.
    
    # Let's try: (0,8), (8,16), (16,23) - main backbone
    # Plus (4,12), (12,20) - intermediate
    # That's 5 edges. Let's check if diameter is <= 4.
    # 0 to 23: 0-8-16-23 = 3 edges. Good.
    # 1 to 22: 1-2-3-4-12-20-21-22 = 7 edges. Bad.
    
    # We need more edges for the "in-between" vertices.
    
    # Let's use a systematic approach:
    # Initial set: for each i in 0..23, add edge (i, i+8) if i+8 < 24
    # That's 16 edges. Then greedily remove.
    
    initial_added = []
    for i in range(16):
        u, v = i, i + 8
        if (u, v) not in edge_set and (v, u) not in edge_set:
            initial_added.append((u, v))
    
    # Also add some (i, i+6) for extra coverage?
    # Let's start with just the +8 edges and see.
    
    # Add them to adjacency
    for u, v in initial_added:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Check if this is valid
    if not is_triangle_free(adj, initial_added):
        # Shouldn't happen with +8 edges on a path (bipartite preserved)
        pass
    
    if not check_diameter(adj, 4):
        # Need more edges. Add (i, i+6) for some i
        extra = []
        for i in range(18):
            u, v = i, i + 6
            if (min(u, v), max(u, v)) not in edge_set:
                extra.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
        initial_added.extend(extra)
    
    # Now do greedy removal
    best_added = initial_added[:]
    improved = True
    while improved and time.time() - t0 < deadline:
        improved = False
        for idx in range(len(best_added)):
            if time.time() - t0 > deadline:
                break
            u, v = best_added[idx]
            # Remove
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard((min(u, v), max(u, v)))
            
            if check_diameter(adj, 4) and is_triangle_free(adj, best_added):
                best_added.pop(idx)
                improved = True
                break
            else:
                # Restore
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
    
    # Convert to required format
    result = []
    for u, v in best_added:
        result.append([int(u), int(v)])
    
    return result
