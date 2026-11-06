import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    # Helper functions using bitmasks
    # adj[i] is an int, bit j set if (i,j) is an edge
    
    def is_triangle_free_adj(adj, u, v):
        # Check if adding (u,v) creates a triangle
        # Need (adj[u] & adj[v]) == 0
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        # Backtracking with bitmasks
        # colors: 0, 1, 2
        # avail[i] is 3-bit mask of available colors
        color = [-1] * n
        avail = [7] * n # 0b111
        
        # Order vertices by degree descending for faster pruning
        # But we need to map back. For n=16, fixed order is fine if optimized.
        # Let's use dynamic variable ordering: pick vertex with fewest available colors.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Find uncolored vertex with minimum available colors
            min_avail = 8
            min_idx = -1
            for i in range(idx, n):
                if color[i] == -1 and avail[i] < min_avail:
                    min_avail = avail[i]
                    min_idx = i
            
            if min_idx == -1:
                return True # All colored
            
            if min_avail == 0:
                return False # Dead end
            
            u = min_idx
            mask = avail[u]
            
            while mask:
                c_bit = mask & (-mask)
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                color[u] = c
                # Update neighbors
                # For each neighbor v, remove color c from avail[v]
                # We need to track changes to undo them.
                # Since n=16, we can just copy avail? Or store changes.
                # Copying list of 16 ints is fast.
                old_avail = avail[:]
                
                neighbors = adj[u]
                m = neighbors
                valid = True
                while m:
                    v_bit = m & (-m)
                    v = v_bit.bit_length() - 1
                    m ^= v_bit
                    if color[v] == -1:
                        new_avail = avail[v] & ~c_bit
                        if new_avail == 0:
                            valid = False
                            break
                        avail[v] = new_avail
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                color[u] = -1
                avail = old_avail
                
            return False
            
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= v_bit
        return edges

    def greedy_complete(adj):
        # Add all possible edges that don't form triangles
        # We want to maximize edges, so we should try to add as many as possible.
        # The order matters. We try multiple random orders.
        possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        
        max_edges = 0
        best_adj = adj[:]
        
        # Try a few random orders
        for _ in range(10):
            current_adj = adj[:]
            random.shuffle(possible)
            for u, v in possible:
                if is_triangle_free_adj(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            
            count = sum(bin(a).count("1") for a in current_adj) // 2
            if count > max_edges:
                max_edges = count
                best_adj = current_adj[:]
        
        return best_adj, max_edges

    # Main loop
    while time.time() < deadline:
        # Generate random triangle-free graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Check if not 3-colorable
        if not is_3_colorable(adj):
            # Complete it
            completed_adj, count = greedy_complete(adj)
            if count > best_count:
                best_count = count
                best_edges = get_edges_from_adj(completed_adj)
        
        # Also try starting with known 4-chromatic graphs
        # e.g., Grötzsch graph embedded in 16 vertices
        # But random generation might find better/denser seeds

    return best_edges
