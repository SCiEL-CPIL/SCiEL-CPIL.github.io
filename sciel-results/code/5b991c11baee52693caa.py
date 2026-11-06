import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 40
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) to graph adj creates a C6.
        A new C6 must use the new edge (u,v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a,b,c,d distinct and not u,v.
        """
        # Use numpy for speed
        adj_u = adj[u]
        adj_v = adj[v]
        
        # a must be a neighbor of u (and not v)
        # d must be a neighbor of v (and not u)
        # We need: a ~ b ~ c ~ d where a in N(u), d in N(v)
        # and a,b,c,d all distinct, none equal to u or v
        
        # For each a in N(u)\{v}:
        #   For each d in N(v)\{u}:
        #     if a == d: continue (need 4 distinct intermediates)
        #     Check if there's a path a-b-c-d of length 3 in the graph
        #     (with b,c distinct from a,d,u,v and from each other)
        
        # More efficient: for each a in N(u)\{v}, for each d in N(v)\{u}, a!=d:
        #   Check if there exist b,c such that a~b, b~c, c~d, all distinct
        #   b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, and b~c, b!=c
        
        # Even more efficient: precompute N(a) ∩ N(d) for relevant pairs
        # But let's just do it carefully with numpy
        
        for a in range(n):
            if a == u or a == v or not adj_u[a]:
                continue
            for d in range(n):
                if d == u or d == v or d == a or not adj_v[d]:
                    continue
                # Check if there's a path a-b-c-d with b,c distinct from {u,v,a,d}
                # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b~c, b!=c
                # Use: (adj[a] & adj[d]) gives common neighbors of a and d
                # But we need b~c where b in N(a) and c in N(d), not necessarily b==c
                # Actually: b in N(a), c in N(d), b~c, b!=c, b,c not in {u,v,a,d}
                
                # Let's iterate over b in N(a)\{u,v,d}
                # For each b, check if any c in N(d)\{u,v,a,b} is adjacent to b
                # This is: adj[b] & adj[d] & ~mask_{u,v,a,b} != 0
                
                # Precompute: for this (a,d) pair, common = adj[a] & adj[d]
                # But that's neighbors of both a and d, which would be b=c case (a triangle a-b-d)
                # We need b~c with b in N(a), c in N(d), b!=c
                # So: for b in N(a)\{u,v,d}: check if adj[b] & adj[d] has any bit set 
                #     in positions not in {u,v,a,b}
                
                # Actually simpler: for b in N(a)\{u,v,d}:
                #   candidates_c = adj[b] & adj[d]
                #   mask out u, v, a, b
                #   if any candidate remains, C6 exists
                
                # To speed up, precompute adj as integer bitmasks
                pass
        
        # Let me redo with bitmasks for speed
        # Convert adj to list of integer bitmasks
        return False
    
    # Better approach: use bitmasks throughout
    # adj_mask[i] = integer with bit j set iff edge (i,j) exists
    
    def create_adj_mask():
        mask = [0] * n
        return mask
    
    def add_edge(mask, u, v):
        mask[u] |= (1 << v)
        mask[v] |= (1 << u)
    
    def check_c6_with_edge(mask, u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks."""
        # Need path u-a-b-c-d-v where a,b,c,d distinct, not u,v
        # a in N(u), d in N(v), a~b~c~d, all distinct
        # For each a in N(u)\{v}:
        #   For each d in N(v)\{u,a}:
        #     Need b,c with a~b, b~c, c~d, b,c not in {u,v,a,d}, b!=c
        #     b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b~c
        #     Equivalent: exists b in N(a)\{u,v,d} such that (mask[b] & mask[d] & ~((1<<u)|(1<<v)|(1<<a)|(1<<b))) != 0
        
        nu = mask[u]
        nv = mask[v]
        
        # a must be in nu, not v
        a_candidates = nu & ~(1 << v)
        d_candidates = nv & ~(1 << u)
        
        while a_candidates:
            a_bit = a_candidates & (-a_candidates)
            a = a_bit.bit_length() - 1
            a_candidates ^= a_bit
            
            # d in d_candidates, d != a
            d_cand = d_candidates & ~(1 << a)
            while d_cand:
                d_bit = d_cand & (-d_cand)
                d = d_bit.bit_length() - 1
                d_cand ^= d_bit
                
                # b in N(a)\{u,v,d}
                na = mask[a]
                b_cand = na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                while b_cand:
                    b_bit = b_cand & (-b_cand)
                    b = b_bit.bit_length() - 1
                    b_cand ^= b_bit
                    
                    # c in N(d)\{u,v,a,b} and c~b
                    # c in (mask[d] & mask[b]) & ~((1<<u)|(1<<v)|(1<<a)|(1<<b))
                    c_cand = (mask[d] & mask[b]) & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                    if c_cand:
                        return True
        
        return False
    
    # Start with 8 copies of K5
    mask = [0] * n
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u, v = 5*k+a, 5*k+b
                mask[u] |= (1 << v)
                mask[v] |= (1 << u)
                edges.append([u, v])
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try adding edges in a smart order
    # Sort candidate edges by some heuristic (e.g., vertices with low degree first?)
    # Or just try all pairs in a fixed order, but use the bitmask check
    
    # Generate all possible edges not yet present
    candidate_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not (mask[i] >> j) & 1:
                candidate_edges.append((i, j))
    
    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(candidate_edges)
    
    # Try adding edges
    for (u, v) in candidate_edges:
        if time.time() > deadline:
            break
        if (mask[u] >> v) & 1:
            continue  # already added
        if not check_c6_with_edge(mask, u, v):
            mask[u] |= (1 << v)
            mask[v] |= (1 << u)
            edges.append([u, v])
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
    
    return best_edges
