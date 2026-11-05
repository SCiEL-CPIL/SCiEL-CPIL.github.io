import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    # Points are 1D subspaces of F_5^3
    # Represented by normalized vectors (first non-zero is 1)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                # Normalize
                for i in range(3):
                    if vec[i] != 0:
                        # Inverse mod 5
                        inv = pow(vec[i], 3, 5)
                        vec = [(v * inv) % 5 for v in vec]
                        break
                tvec = tuple(vec)
                if tvec not in seen:
                    seen.add(tvec)
                    points.append(tvec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # 2. Construct polarity graph
    # Edge if dot product is 0 mod 5
    adj = [0] * n  # Bitmasks
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback: if construction is wrong, use a random sparse graph?
        # But PG(2,5) should be correct.
        # For safety, if not 90, we just proceed and adjust later.
        pass

    # Helper to count C4s using bitmasks
    def count_c4s(adj_list):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                # Count bits
                k = bin(common).count('1')
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    current_c4s = count_c4s(adj)
    
    # 3. Add the 91st edge
    if len(edges) < 91:
        best_delta = float('inf')
        best_edge = None
        
        # Precompute neighbors for speed? 
        # With bitmasks, we can just iterate.
        
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    # Calculate C4 increment
                    # C4s created by adding (i,j) is number of paths of length 3 between i and j
                    # Path i-x-y-j
                    # Sum over x in N(i) of |N(x) intersect N(j)|
                    # Since graph is C4-free, |N(x) intersect N(j)| is 0 or 1?
                    # Actually, if there are 2, it's a C4 in the old graph.
                    # So in a C4-free graph, any pair has at most 1 common neighbor.
                    # So increment is sum_{x in N(i)} (1 if x and j have common neighbor else 0)
                    # Wait, if x and j have a common neighbor y, then i-x-y-j is a path.
                    # Adding (i,j) makes i-j-y-x-i a C4.
                    
                    increment = 0
                    # Iterate neighbors of i
                    neighbors_i = adj[i]
                    while neighbors_i:
                        lsb = neighbors_i & (-neighbors_i)
                        x = lsb.bit_length() - 1
                        neighbors_i ^= lsb
                        
                        # Check if x and j have common neighbor
                        # common = adj[x] & adj[j]
                        # if common != 0: increment += 1
                        # Actually, we just need to know if there is ANY common neighbor.
                        # But wait, if there are multiple, they form multiple C4s?
                        # No, in a C4-free graph, there is at most 1 common neighbor for any pair.
                        # So increment += 1 if (adj[x] & adj[j]) != 0 else 0
                        
                        if (adj[x] & adj[j]) != 0:
                            increment += 1
                            
                    if increment < best_delta:
                        best_delta = increment
                        best_edge = (i, j)
                        
        if best_edge:
            i, j = best_edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges.append((i, j))
            current_c4s = count_c4s(adj)
            
    # 4. Local Search
    # Try swaps: remove (u,v), add (x,y)
    # If time permits, do full sweep.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap = None
        best_c4 = current_c4s
        
        # To speed up, we can limit the search space or use random sampling if too slow
        # But n=31 is small.
        # We will iterate through all edges to remove.
        # For each, iterate through all non-edges to add.
        
        # Optimization: 
        # We can precompute the list of edges and non-edges.
        
        edges_list = list(edges)
        # Non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
                    
        # If we have many non-edges, this loop is large.
        # m=91, non-m=374. 91*374 ~ 34k swaps.
        # Each swap: 2 updates, 1 count_c4s.
        # count_c4s is O(n^2) ~ 1000 ops.
        # 34k * 1000 = 34 million ops.
        # Python can do ~10-50 million simple ops per second.
        # So ~1-3 seconds per full sweep.
        # We can do multiple sweeps.
        
        for e1 in edges_list:
            u1, v1 = e1
            # Remove e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            # Find best non-edge to add
            local_best_c4 = float('inf')
            local_best_edge = None
            
            for e2 in non_edges:
                u2, v2 = e2
                # Skip if e2 is e1 (should not be in non_edges)
                # Add e2
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                c4 = count_c4s(adj)
                
                # Revert add
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                
                if c4 < local_best_c4:
                    local_best_c4 = c4
                    local_best_edge = e2
                    
            # Revert remove
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            
            if local_best_edge and local_best_c4 < best_c4:
                best_c4 = local_best_c4
                best_swap = (e1, local_best_edge)
                
        if best_swap:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Perform swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Update edges list
            edges.remove(e1)
            edges.append(e2)
            
            current_c4s = best_c4
            improved = True
            
    # Return edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                result.append([i, j])
                
    return result
