import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Keep 10s margin
    
    n = 20
    # Precompute all pairs
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edge_count = 0
    best_adj = None
    
    # Helper to count bits
    def popcount(x):
        return x.bit_count()
        
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) to graph represented by adj list of bitmasks
        violates the condition that no 3 vertices have 3 common neighbors.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # Extract neighbors as lists for iteration
        # Since n=20, this is cheap
        nu_list = [i for i in range(n) if (Nu >> i) & 1]
        nv_list = [i for i in range(n) if (Nv >> i) & 1]
        
        # Check triples involving v and two neighbors of u
        # If u is added to N(v), then for any a,b in N(u), 
        # u becomes a common neighbor of v, a, b if u is adjacent to a and b.
        # We need to ensure that before adding u, the number of common neighbors
        # of v, a, b is not already 2.
        for i in range(len(nu_list)):
            a = nu_list[i]
            for j in range(i + 1, len(nu_list)):
                b = nu_list[j]
                # Common neighbors of v, a, b
                common = Nv & adj[a] & adj[b]
                if popcount(common) == 2:
                    return False
        
        # Check triples involving u and two neighbors of v
        for i in range(len(nv_list)):
            a = nv_list[i]
            for j in range(i + 1, len(nv_list)):
                b = nv_list[j]
                # Common neighbors of u, a, b
                common = Nu & adj[a] & adj[b]
                if popcount(common) == 2:
                    return False
            
        return True

    while time.time() < deadline:
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if (adj[u] >> v) & 1:
                continue
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Count edges
        count = 0
        for i in range(n):
            count += popcount(adj[i])
        count //= 2
        
        if count > best_edge_count:
            best_edge_count = count
            best_adj = adj[:]
            
            # Optional: Early stop if we reach a very high number?
            # The hint says ~100 is possible. Max possible is 190.
            # If we hit 100+, we can keep searching for more.
            
    # Convert best_adj back to list of edges
    if best_adj is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result_edges.append([i, j])
                
    return result_edges
