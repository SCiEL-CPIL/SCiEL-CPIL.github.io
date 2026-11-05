import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    # Map points to indices 0..30
    point_index = {p: i for i, p in enumerate(points)}
    
    # 2. Construct adjacency matrix for polarity graph
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Dot product mod 5
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Verify edge count
    edge_count = np.sum(adj) // 2
    # Should be 90
    
    # 3. Find best edge to add
    best_delta = float('inf')
    best_edge = None
    
    # Precompute neighbor lists for speed
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                # Count paths of length 3 between i and j
                # Sum over w in N(i) of |N(w) intersect N(j)|
                count = 0
                for w in neighbors[i]:
                    # Check intersection of neighbors[w] and neighbors[j]
                    # Using sets for intersection might be slow, but n=31 is small
                    # Or just loop
                    # Since degree is ~6, this is very fast
                    for x in neighbors[w]:
                        if adj[x, j] == 1:
                            count += 1
                if count < best_delta:
                    best_delta = count
                    best_edge = (i, j)
                    
    # Add the best edge
    if best_edge:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
        
    # 4. Local Search
    # Current C4 count
    def count_c4s(adj):
        # Number of C4s is (1/8) * sum_{i,j} (A^2_{ij})^2 - ... ?
        # Simpler: count pairs of neighbors for each pair of vertices
        # For each pair (u, v), let k = |N(u) intersect N(v)|
        # Each C4 contributes to 2 pairs of opposite vertices?
        # A C4 is u-v-w-x-u.
        # Pairs of opposite vertices: (u,w) and (v,x).
        # For (u,w), common neighbors are v and x. So k >= 2.
        # Number of C4s = sum_{u<v} C(k_{uv}, 2) / 2 ?
        # Let's verify.
        # A C4 has 2 pairs of opposite vertices.
        # For each pair of opposite vertices, they have exactly 2 common neighbors (the other two vertices of the C4).
        # So each C4 is counted twice in the sum of C(k, 2) over all pairs?
        # If a pair has k common neighbors, it forms C(k, 2) C4s?
        # Yes, any 2 common neighbors form a C4 with the pair.
        # So total C4s = (1/2) * sum_{u<v} C(|N(u) \cap N(v)|, 2).
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.intersect1d(neighbors[u], neighbors[v])
                k = len(common)
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2
        
    current_c4s = count_c4s(adj)
    
    # Try to improve
    # We have 91 edges.
    # List of edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_score = current_c4s
        best_swap = None
        
        # Try swapping each edge with each non-edge
        # This is O(m * (n^2-m)) * cost
        # m=91, non-edges=374. ~34k swaps.
        # Cost of swap: update adj, recount C4s?
        # Recounting is O(n^2 * deg^2) ~ 31^2 * 36 ~ 34k.
        # Total ~ 10^9 ops. Too slow in Python.
        
        # Optimized:
        # Only try swaps that might reduce C4s.
        # Or use a faster C4 counter.
        
        # Given the small size, we can just try a limited number of random swaps or greedy.
        # Since we start with 1 or 2, and min is 1, we might already be optimal.
        # Let's just do a few random restarts or local moves if time permits.
        
        # For simplicity and safety, if current_c4s <= 1, we can stop.
        if current_c4s <= 1:
            break
            
        # Random local search
        for _ in range(100): # limited iterations
            if time.time() > deadline - 5:
                break
            # Pick a random edge to remove
            e_idx = random.randint(0, len(edges)-1)
            u, v = edges[e_idx]
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Pick a random non-edge to add
            # Find a non-edge
            # Simple way: pick random i, j
            non_edge_found = False
            while not non_edge_found:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j and adj[i, j] == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    non_edge_found = True
                    
            new_c4s = count_c4s(adj)
            
            if new_c4s < current_c4s:
                current_c4s = new_c4s
                edges[e_idx] = (i, j)
                improved = True
            else:
                # Revert
                adj[i, j] = 0
                adj[j, i] = 0
                adj[u, v] = 1
                adj[v, u] = 1
                
    # Return edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result
