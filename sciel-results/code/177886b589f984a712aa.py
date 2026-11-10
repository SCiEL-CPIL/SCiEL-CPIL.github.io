import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n = 31
    target_edges = 91
    
    # Build polarity graph of PG(2,5) as initial structure
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Precompute neighbor lists for efficiency
    def get_neighbors(i):
        return np.where(adj[i] > 0)[0]
    
    # Count C4s using matrix operations
    def count_c4s(adj_matrix):
        # A^2[i,j] = number of common neighbors of i and j
        A2 = adj_matrix @ adj_matrix
        
        # For each pair (i,j) that is NOT an edge, add C(A2[i,j], 2)
        # C(k,2) = k*(k-1)/2
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i,j] == 0:
                    k = A2[i,j]
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    current_c4s = count_c4s(adj)
    
    # Helper to count new C4s when adding edge (u, v)
    def count_new_c4s_add(u, v, adj_matrix):
        # When adding edge (u,v), new C4s are those that use this edge
        # A C4 using edge (u,v) has form u-v-a-b-u where a~v, b~u, a~b, a!=b, a!=u, b!=v
        # This equals the number of common neighbors of u and v that are connected
        # Actually: new C4s = number of pairs (a,b) where a~v, b~u, a~b, a!=b
        # = (A2_new[u,v] - 1) choose 2 - ... 
        # Simpler: just count paths of length 2 from u to v through existing edges
        # that form a C4 with the new edge
        # A C4 u-v-a-b-u requires: v~a, a~b, b~u, and a!=b
        # So new C4s = sum over a in N(v)\{u} of |N(a) ∩ N(u)\{v,a}|
        
        # Using matrix: this is related to A2 but we need to be careful
        # Let's compute directly
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if adj_matrix[v, a] == 1:
                for b in range(n):
                    if b == u or b == v or b == a:
                        continue
                    if adj_matrix[u, b] == 1 and adj_matrix[a, b] == 1:
                        count += 1
        return count
    
    # Helper to count C4s removed when removing edge (u, v)
    def count_c4s_removed(u, v, adj_matrix):
        # C4s that use edge (u,v)
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if adj_matrix[v, a] == 1:
                for b in range(n):
                    if b == u or b == v or b == a:
                        continue
                    if adj_matrix[u, b] == 1 and adj_matrix[a, b] == 1:
                        count += 1
        return count
    
    # Greedy optimization: add edges to reach 91, minimizing C4s
    while np.sum(adj) // 2 < target_edges:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_cost = float('inf')
        
        # Try all non-edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    cost = count_new_c4s_add(i, j, adj)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
    
    # If we have too many edges, remove greedily
    while np.sum(adj) // 2 > target_edges:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_saving = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    saving = count_c4s_removed(i, j, adj)
                    if saving > best_saving:  # Remove edge that removes most C4s
                        best_saving = saving
                        best_edge = (i, j)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u, v] = 0
        adj[v, u] = 0
    
    # Local search: try edge swaps to reduce C4s
    # Swap: remove edge (u,v) and add edge (x,y) if it reduces C4s
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_c4s = count_c4s(adj)
        
        # Try all possible swaps
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
        non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == 0]
        
        for (u, v) in edges_list:
            if time.time() > deadline:
                break
            for (x, y) in non_edges_list:
                # Skip if swap doesn't change edge count or creates duplicate
                if (u, v) == (x, y) or (u, v) == (y, x):
                    continue
                # Check if x,y are not already connected to create issues
                # Try the swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                new_c4s = count_c4s(adj)
                
                if new_c4s < current_c4s:
                    current_c4s = new_c4s
                    improved = True
                    # Keep the swap
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
        
        if time.time() > deadline:
            break
    
    # Extract edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    return result
