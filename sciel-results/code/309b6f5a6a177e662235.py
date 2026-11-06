"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 40
    
    def bitmask_c6_free(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6 using bitmask operations.
        adj_bits[i] is an integer bitmask of neighbors of vertex i.
        """
        if (adj_bits[u] >> v) & 1:
            return True  # already an edge
        
        # Neighbors of u (excluding v) and neighbors of v (excluding u)
        nu = adj_bits[u] & ~(1 << v)
        nv = adj_bits[v] & ~(1 << u)
        
        # For each a in N(u), d in N(v), check if there's a path a-b-c-d of length 3
        # in the graph without using u or v as internal vertices.
        # Path: a - b - c - d where a~u, d~v, b~a, c~d, b!=c, and none of a,b,c,d is u or v.
        
        # Efficient approach: 
        # For each pair (a, d) with a in N(u), d in N(v):
        #   Check if there exists b ~ a and c ~ d such that b ~ c, 
        #   with b != c, b not in {u,v,d}, c not in {u,v,a}, a != d.
        
        # Even more efficient: precompute for each vertex the set of vertices at distance 2.
        # dist2[x] = union of N(y) for y in N(x), minus N(x) itself and x.
        # Then a path a-b-c-d exists iff there's some c in N(d) such that c is in dist2_from_a_restricted.
        
        # Let's use a different approach: 
        # A C6 containing edge (u,v) has form u-a-b-c-d-v-u.
        # So we need: a in N(u), d in N(v), and a path a-b-c-d of length 3 in G\{u,v}.
        
        # For each a in N(u) and d in N(v) with a != d:
        #   Check if there exist b, c with b in N(a), c in N(d), b in N(c), 
        #   b != c, b not in {u,v}, c not in {u,v}, and b != d, c != a.
        
        # Let's iterate over a in N(u), d in N(v):
        a_mask = nu
        d_mask = nv
        
        # For efficiency, iterate over set bits
        # Get list of a's and d's
        a_list = [i for i in range(n) if (a_mask >> i) & 1]
        d_list = [i for i in range(n) if (d_mask >> i) & 1]
        
        for a in a_list:
            if a in d_list:
                continue  # a == d would create a shorter cycle
            # N(a) excluding u, v, and we want b such that b ~ c for some c in N(d)
            na = adj_bits[a] & ~(1 << u) & ~(1 << v)
            nd = adj_bits[d] & ~(1 << u) & ~(1 << v)
            
            # We need b in N(a) \ {u,v} and c in N(d) \ {u,v} such that b ~ c, b != c
            # b ~ c means c in N(b)
            # For each b in N(a)\{u,v}, check if N(b) intersects N(d)\{u,v}\{b}
            b_mask = na
            # c candidates: N(d) \ {u,v}
            c_mask = nd
            
            # Check intersection: for each b, see if N(b) & c_mask has any bit other than b itself
            # Actually, b ~ c means c is in N(b). We need c in c_mask and c in N(b), c != b.
            # So: N(b) & c_mask should have a bit set that is not b.
            
            # Efficient: compute union of N(b) for b in b_mask, intersect with c_mask
            # But that's expensive. Instead, iterate over smaller set.
            
            # Let's just check: is there any edge between N(a)\{u,v} and N(d)\{u,v} 
            # where the endpoints are distinct?
            
            # Precompute: for the current graph, we can check edge existence quickly.
            # For b in N(a)\{u,v}, c in N(d)\{u,v}, b!=c: check if adj_bits[b] has bit c.
            
            # To speed up: if |N(a)\{u,v}| * |N(d)\{u,v}| is small, just check all pairs.
            # Otherwise, use a smarter method.
            
            # Let's use: for each b in the smaller of (N(a)\{u,v}, N(d)\{u,v}), 
            # check if N(b) intersects the other set.
            
            na_list = [i for i in range(n) if (na >> i) & 1]
            nd_list = [i for i in range(n) if (nd >> i) & 1]
            
            if len(na_list) <= len(nd_list):
                for b in na_list:
                    # Check if N(b) has any vertex in nd_list that is not b
                    nb = adj_bits[b]
                    # Intersect nb with nd_mask, exclude b
                    inter = nb & nd_mask & ~(1 << b)
                    if inter != 0:
                        return False  # C6 found
            else:
                for c in nd_list:
                    # Check if N(c) has any vertex in na_list that is not c
                    nc = adj_bits[c]
                    inter = nc & na_mask & ~(1 << c)
                    if inter != 0:
                        return False  # C6 found
        
        return True  # No C6 found
    
    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += bin(adj_bits[i]).count('1')
        return total // 2
    
    # Start with a random graph and try to add edges
    # Or start from a dense-ish structure and refine
    
    # Strategy: Start with 8 copies of K5 (the seed), then try to add cross-edges
    # Also try random restarts with different initial structures
    
    best_edges = []
    best_count = 0
    
    # Seed: 8 copies of K5
    adj_bits = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                adj_bits[base + a] |= (1 << (base + b))
                adj_bits[base + b] |= (1 << (base + a))
    
    current_count = count_edges(adj_bits)
    best_count = current_count
    best_adj = adj_bits[:]
    
    # Try to add edges greedily
    # Generate all non-edges in random order
    all_non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not ((adj_bits[u] >> v) & 1):
                all_non_edges.append((u, v))
    
    random.shuffle(all_non_edges)
    
    # Greedy: try to add each non-edge if it doesn't create a C6
    for (u, v) in all_non_edges:
        if time.time() > deadline:
            break
        if bitmask_c6_free(adj_bits, u, v):
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            current_count += 1
    
    if current_count > best_count:
        best_count = current_count
        best_adj = adj_bits[:]
    
    # Try multiple random restarts with different orderings
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Reset to seed
        adj_bits = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    adj_bits[base + a] |= (1 << (base + b))
                    adj_bits[base + b] |= (1 << (base + a))
        
        current_count = count_edges(adj_bits)
        
        all_non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_bits[u] >> v) & 1):
                    all_non_edges.append((u, v))
        
        random.shuffle(all_non_edges)
        
        for (u, v) in all_non_edges:
            if time.time() > deadline:
                break
            if bitmask_c6_free(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_bits[:]
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
