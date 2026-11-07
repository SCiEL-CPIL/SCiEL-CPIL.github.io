"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # The base graph is a path on 24 vertices: 0-1-2-...-23
    # The path is bipartite with partitions:
    # A = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22} (even)
    # B = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23} (odd)
    # To remain triangle-free, we can only add edges between A and B.
    #
    # We need to minimize the number of added edges while ensuring:
    # 1. The graph remains connected (already connected, adding edges preserves this)
    # 2. The diameter is at most 4
    #
    # Strategy: Use a greedy approach or a smart constructive solution.
    #
    # Key insight: If we can find a small set of "hub" vertices such that
    # every other vertex is within distance 2 of some hub, and hubs are within
    # distance 2 of each other, then the diameter will be at most 4.
    #
    # Let's try a constructive approach with a fixed set of edges.
    # 
    # Consider connecting every 4th vertex to create shortcuts:
    # Add edges: (0,4), (4,8), (8,12), (12,16), (16,20), (20,24) - but 24 doesn't exist
    # 
    # Let's think more carefully. The path has length 23.
    # With diameter 4, we need to be able to reach any vertex from any other in 4 steps.
    #
    # A good strategy: add edges that create a "ladder" or skip patterns.
    # 
    # Let's try: for each i, connect i to i+4 if they are in different partitions.
    # i and i+4 have the same parity, so they're in the same partition. Can't connect.
    #
    # Connect i to i+3? Different parity. 
    # Let's add edges (i, i+3) for i = 0, 1, 2, ... 
    # But we need to be careful about the count.
    #
    # Actually, let me use a simpler constructive solution:
    # Add edges to create a structure where the effective path length is reduced.
    #
    # Here's a known good solution pattern:
    # Connect pairs that are 4 apart in the path but in different partitions.
    # Since (i, i+4) are same partition, use (i, i+3) or (i, i+5).
    #
    # Let me try a different approach: greedy with backtracking for a small number of edges.
    # Given the constraint of 60s, and the fact that we just need a valid solution,
    # let me hardcode a verified solution.
    
    # Verified solution: Add 12 edges
    # Pattern: connect i to i+4 for even i (these are A-to-A, invalid)
    # Connect i to i+3 for all i where i and i+3 have different parity (always true since 3 is odd)
    
    # Let's use a systematic construction:
    # Add edges (i, i+4) is invalid (same partition)
    # Add edges (i, i+5) - different partition since 5 is odd
    
    # Actually, the simplest valid high-quality solution:
    # Add 12 edges: (i, i+4) won't work. 
    # Let's add (i, i+6) for i=0,1,...,17? No, 6 is even, same partition.
    
    # (i, i+3): 3 is odd, so different partition. Valid.
    # Let's add a sparse set of (i, i+3) edges to reduce diameter.
    
    # Add edges: (0,3), (3,6), (6,9), (9,12), (12,15), (15,18), (18,21)
    # That's 7 edges, creating a fast path through odd vertices.
    # And: (1,4), (4,7), (7,10), (10,13), (13,16), (16,19), (19,22)
    # That's 7 more edges, creating a fast path through even vertices.
    # Total: 14 edges.
    
    # Let me verify this works and count more carefully.
    # With these edges, can we reach any node in 4 steps?
    # From 0: 0->3->6->9->12 (4 steps to 12), or 0->1->4->7->10 (4 steps to 10)
    # From 0 to 23: 0->3->6->9->12->15->18->21->23... that's too many.
    # 0->3 (1), 3->6 (2), 6->9 (3), 9->12 (4)... need to reach 23.
    # 23 is odd. 21 is odd, connected to 19 and 22 and 23.
    # 0->3->6->9->12 (4 steps). From 12, 12->15->18->21->23 is 4 more. Total 8. Too many.
    
    # Need denser connections. Let me just use a greedy approach with verification.
    
    import time
    deadline = time.time() + 50  # safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(e))
        base_edges.add(tuple(reversed(e)))
    
    # Adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Partitions
    A = set(i for i in range(n) if i % 2 == 0)
    B = set(i for i in range(n) if i % 2 == 1)
    
    def get_diameter():
        # BFS from each node
        max_d = 0
        for start in range(n):
            dist = {start: 0}
            queue = [start]
            while queue:
                next_q = []
                for u in queue:
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            next_q.append(v)
                queue = next_q
            if len(dist) < n:
                return float('inf')
            max_d = max(max_d, max(dist.values()))
            if max_d > 4:
                break
        return max_d
    
    def is_triangle_free():
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    added_edges = []
    
    # Greedy: repeatedly find the pair with maximum distance, add an edge that reduces it most
    # Candidate edges: (u,v) where u in A, v in B, not in base, not already added
    candidate_pool = []
    for u in A:
        for v in B:
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidate_pool.append((u, v))
    
    # Sort candidates by how much they might help (heuristic: shorter "detour" reduction)
    # Just try them in order of proximity first for efficiency
    
    for _ in range(30):  # try up to 30 edges
        if time.time() > deadline:
            break
        
        d = get_diameter()
        if d <= 4:
            break
        if d == float('inf'):
            break
        
        # Find the pair with max distance
        # For efficiency, just check a few random pairs or the ends
        max_d_pair = None
        max_dist = 0
        # Check distances from a few key nodes
        for start in [0, n-1, n//2]:
            dist = {start: 0}
            queue = [start]
            while queue:
                next_q = []
                for u in queue:
                    for v in adj[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            next_q.append(v)
                queue = next_q
            for v, d_val in dist.items():
                if d_val > max_dist:
                    max_dist = d_val
                    max_d_pair = (start, v)
        
        if max_d_pair is None:
            break
        
        u_target, v_target = max_d_pair
        
        # Try to add an edge that brings u_target and v_target closer
        # Add edge from u_target to some neighbor of v_target, or vice versa
        # Or add a direct shortcut
        
        best_edge = None
        best_score = -1
        
        for (u, v) in candidate_pool:
            if (u, v) in base_edges or (v, u) in base_edges:
                continue
            # Check if adding this edge reduces the distance between u_target and v_target
            # Simple heuristic: if u is closer to u_target and v is closer to v_target
            # Just pick the edge that connects a node near u_target to a node near v_target
            
            # Compute distance from u to u_target in current graph
            # This is expensive; use a proxy: path distance
            # u_target and u are both in same or different partition
            # Just pick edges that are "in between"
            
            # Simpler: just add edges that are far apart in the path to create shortcuts
            path_dist = abs(u - v)
            if path_dist >= 3 and path_dist <= 7:
                score = path_dist  # prefer longer shortcuts
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
        
        if best_edge is None:
            # Try any valid edge
            for (u, v) in candidate_pool:
                if (u, v) not in base_edges and (v, u) not in base_edges:
                    best_edge = (u, v)
                    break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        # Add the edge
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])
        candidate_pool.remove((u, v))
        
        # Verify triangle-free
        if not is_triangle_free():
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)
            added_edges.pop()
            continue
    
    # Final check
    d = get_diameter()
    if d > 4 or not is_triangle_free():
        # Fallback: add more edges greedily
        for (u, v) in candidate_pool:
            if time.time() > deadline:
                break
            if (u, v) in base_edges or (v, u) in base_edges:
                continue
            adj[u].add(v)
            adj[v].add(u)
            added_edges.append([u, v])
            if get_diameter() <= 4 and is_triangle_free():
                break
    
    # Ensure proper types
    result = []
    for e in added_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
