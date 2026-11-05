import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 40
    
    # Initialize with 8 disjoint K5s
    # adj[i] is a bitset of neighbors of i
    adj = [0] * n
    edges = []
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    
    def find_c6_path(u, v):
        """
        Returns a list [a, b, c, d] such that u-a-b-c-d-v is a path of length 5,
        or None if no such path exists.
        Assumes (u, v) is not an edge.
        """
        nu = adj[u]
        nv = adj[v]
        
        # Iterate a in N(u)
        a_bits = nu
        while a_bits:
            lsb_a = a_bits & -a_bits
            a = lsb_a.bit_length() - 1
            a_bits -= lsb_a
            
            # b in N(a) \ {u}
            # Note: adj[a] contains u if edge exists. We are checking if adding (u,v) creates C6.
            # The path uses existing edges. So u is a neighbor of a.
            # We need b != u.
            b_bits = adj[a] & ~lsb_a
            while b_bits:
                lsb_b = b_bits & -b_bits
                b = lsb_b.bit_length() - 1
                b_bits -= lsb_b
                
                # c in N(b) \ {a}
                c_bits = adj[b] & ~lsb_a
                while c_bits:
                    lsb_c = c_bits & -c_bits
                    c = lsb_c.bit_length() - 1
                    c_bits -= lsb_c
                    
                    # d in N(c) \ {b} ∩ N(v)
                    d_bits = (adj[c] & ~lsb_b) & nv
                    if d_bits:
                        # Found a path
                        # d is any bit in d_bits
                        lsb_d = d_bits & -d_bits
                        d = lsb_d.bit_length() - 1
                        return [a, b, c, d]
        return None

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Remove from edges list efficiently? 
        # For simplicity, we might not maintain the list perfectly during swaps,
        # but we need to return the final list.
        # We can just rebuild the list at the end or maintain a set.
        # Let's maintain a set of tuples for O(1) lookup, and rebuild list at end.
        pass

    # Use a set for current edges for fast lookup
    current_edges = set(map(tuple, edges))
    
    # Precompute all non-edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    non_edges = [p for p in all_pairs if p not in current_edges]
    
    # Greedy phase: add as many edges as possible
    # Shuffle non_edges to avoid bias
    random.shuffle(non_edges)
    
    i = 0
    while i < len(non_edges):
        if time.time() > deadline - 10: # Reserve time for local search
            break
        u, v = non_edges[i]
        if (u, v) in current_edges:
            i += 1
            continue
        path = find_c6_path(u, v)
        if path is None:
            add_edge(u, v)
            current_edges.add((u, v))
            # Remove from non_edges? We can just skip if it's in current_edges
        i += 1
        
    # Local search phase
    # Strategy: Try to add a random non-edge. If it fails, remove an edge from the blocking path.
    # This increases edge count by 0 (remove 1, add 1) but changes structure.
    # We want to maximize edge count. So we prefer adding edges.
    # If we can't add, we swap.
    
    # Let's restart the non_edges list with the current state
    non_edges = [p for p in all_pairs if p not in current_edges]
    
    while time.time() < deadline:
        if not non_edges:
            break
        u, v = random.choice(non_edges)
        if (u, v) in current_edges:
            continue
            
        path = find_c6_path(u, v)
        if path is None:
            add_edge(u, v)
            current_edges.add((u, v))
            # Remove (u,v) from non_edges? 
            # It's a list, so we can't easily remove. We'll just check membership.
        else:
            # Remove a random edge from the path
            # Path is u-a-b-c-d-v
            # Edges in path: (u,a), (a,b), (b,c), (c,d), (d,v)
            path_edges = [(u, path[0]), (path[0], path[1]), (path[1], path[2]), (path[2], path[3]), (path[3], v)]
            # Ensure edges are stored as (min, max)
            path_edges = [tuple(sorted(e)) for e in path_edges]
            
            # Pick a random edge from the path to remove
            rem_u, rem_v = random.choice(path_edges)
            
            # Remove it
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            current_edges.discard((rem_u, rem_v))
            
            # Try to add (u, v)
            # It might still fail if there's another C6 path
            path2 = find_c6_path(u, v)
            if path2 is None:
                add_edge(u, v)
                current_edges.add((u, v))
            else:
                # If it still fails, put the removed edge back
                # and try a different strategy or just skip
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                current_edges.add((rem_u, rem_v))

    # Reconstruct edges list from adj
    final_edges = []
    for u in range(n):
        bits = adj[u]
        while bits:
            lsb = bits & -bits
            v = lsb.bit_length() - 1
            if u < v:
                final_edges.append([u, v])
            bits -= lsb
            
    return final_edges
