import random
import time

def construct():
    n = 40
    best_edges = set()
    best_count = 0
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Bitmask helpers
    def mask_of(*vertices):
        m = 0
        for v in vertices:
            m |= (1 << v)
        return m

    # Precompute single-bit masks for speed?
    bit_masks = [1 << i for i in range(n)]
    
    def has_c6_with_edge(u, v, adj_masks):
        # Check if path u-a-b-c-d-v exists in G \ {(u,v)}
        # adj_masks[x] is bitmask of neighbors of x
        # We assume (u,v) is NOT in adj_masks yet (we check before adding)
        
        # a in N(u) \ {v}
        # b in N(a) \ {u}
        # c in N(b) \ {a}
        # d in N(c) \ {b} AND d in N(v) \ {u,a,b,c,v}
        
        # Iterate a
        mask_u = adj_masks[u]
        # Clear v from mask_u to get candidates for a
        mask_a_candidates = mask_u & ~bit_masks[v]
        
        while mask_a_candidates:
            # Pick a
            a_bit = mask_a_candidates & -mask_a_candidates
            a = a_bit.bit_length() - 1
            mask_a_candidates ^= a_bit
            
            # b in N(a) \ {u}
            mask_b_candidates = adj_masks[a] & ~bit_masks[u]
            
            while mask_b_candidates:
                b_bit = mask_b_candidates & -mask_b_candidates
                b = b_bit.bit_length() - 1
                mask_b_candidates ^= b_bit
                
                # c in N(b) \ {a}
                mask_c_candidates = adj_masks[b] & ~bit_masks[a]
                
                while mask_c_candidates:
                    c_bit = mask_c_candidates & -mask_c_candidates
                    c = c_bit.bit_length() - 1
                    mask_c_candidates ^= c_bit
                    
                    # d in N(c) \ {b} AND d in N(v)
                    # d must be in N(c) and N(v)
                    # d must not be u, a, b, c, v
                    mask_d_candidates = adj_masks[c] & adj_masks[v]
                    # Exclude b, c, v, u, a
                    # Note: c is in N(c)? No, simple graph.
                    # b is in N(c)? Yes, we iterated b from N(a) but b is neighbor of c? 
                    # Wait, b is neighbor of c? 
                    # Path: a-b-c. So b-c is an edge.
                    # So b is in N(c).
                    # We need d != b.
                    # d != c (c not in N(c))
                    # d != v (v not in N(v))
                    # d != u, a
                    
                    mask_exclude = bit_masks[u] | bit_masks[a] | bit_masks[b] | bit_masks[c] | bit_masks[v]
                    mask_d_candidates &= ~mask_exclude
                    
                    if mask_d_candidates:
                        return True
        
        return False

    def remove_all_c6s(edges, adj_masks):
        # Greedy removal: if C6 exists, remove a random edge from it?
        # Or just check all edges and remove if adding it creates C6? 
        # Better: iterate edges, if edge is part of a C6, remove it.
        # To find if edge (u,v) is in C6, we check has_c6_with_edge(u,v, adj_masks without u-v)
        # This is expensive if we do it for all edges.
        # Alternative: Start with random graph, then try to add edges.
        # If we start with 0 edges, we don't need removal.
        # But we want to start dense to get high count.
        # Let's start with 0 and add greedily.
        pass

    for seed in range(100):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        edges = set()
        adj_masks = [0] * n
        
        # Initialize with a random sparse graph to give structure?
        # Or start empty.
        # Starting empty and adding greedily is safe.
        
        # To get better results, we can start with a known good structure?
        # Let's try starting with 0 and adding.
        
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if (u, v) in edges or (v, u) in edges:
                continue
            if has_c6_with_edge(u, v, adj_masks):
                continue
            # Add edge
            edges.add((u, v))
            adj_masks[u] |= bit_masks[v]
            adj_masks[v] |= bit_masks[u]
            
        # Now try to add more edges that were previously blocked?
        # The greedy pass might have blocked edges due to order.
        # Try a second pass with shuffled order.
        random.shuffle(all_edges)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for u, v in all_edges:
                if (u, v) in edges:
                    continue
                if has_c6_with_edge(u, v, adj_masks):
                    continue
                edges.add((u, v))
                adj_masks[u] |= bit_masks[v]
                adj_masks[v] |= bit_masks[u]
                improved = True
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return [list(e) for e in best_edges]
