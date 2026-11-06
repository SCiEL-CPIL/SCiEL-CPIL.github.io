"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    n = 40
    
    # Use bitsets for adjacency for O(1) intersection checks
    adj = [0] * n
    edges = []
    
    # Start with 8 disjoint K5s (80 edges, no C6)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    def would_create_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) is u-a-b-c-d-v-u.
        This requires a path of length 5 from u to v: u-a-b-c-d-v.
        a in N(u), b in N(a), c in N(b), d in N(c), and d in N(v).
        All vertices u,v,a,b,c,d must be distinct.
        """
        # Iterate a in N(u)
        a_mask = adj[u]
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # b in N(a) \ {u, v, a}
            b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            
            while b_mask:
                lsb_b = b_mask & -b_mask
                b = lsb_b.bit_length() - 1
                b_mask ^= lsb_b
                
                # c in N(b) \ {a, u, v, b}
                c_mask = adj[b] & ~(1 << a) & ~(1 << u) & ~(1 << v) & ~(1 << b)
                
                while c_mask:
                    lsb_c = c_mask & -c_mask
                    c = lsb_c.bit_length() - 1
                    c_mask ^= lsb_c
                    
                    # d must be in N(c) AND N(v)
                    # d must be distinct from a, b, c, u, v
                    d_candidates = adj[c] & adj[v]
                    # Exclude a, b, c, u, v
                    d_candidates &= ~(1 << a)
                    d_candidates &= ~(1 << b)
                    d_candidates &= ~(1 << c)
                    d_candidates &= ~(1 << u)
                    d_candidates &= ~(1 << v)
                    
                    if d_candidates:
                        return True
        
        return False

    # Generate list of all possible edges not in initial graph
    all_possible = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                all_possible.append((i, j))
    
    random.shuffle(all_possible)
    
    # Greedy addition with randomization
    for pass_num in range(50):
        if time.time() > deadline:
            break
        random.shuffle(all_possible)
        changed = False
        for (u, v) in all_possible:
            if time.time() > deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if not would_create_c6(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
        if not changed:
            pass
    
    return best_edges
# EVOLVE-BLOCK-END
