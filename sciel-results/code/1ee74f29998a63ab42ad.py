import time
import random

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.append([u, v])
                m ^= lsb
        return edges

    def make_mycielski(base_adj, base_n, new_n):
        # base_adj is for vertices 0..base_n-1
        # new_n is the total number of vertices
        # Copies are base_n .. base_n + base_n - 1
        # w is base_n + base_n
        adj = [0] * new_n
        
        # Copy base graph
        for u in range(base_n):
            adj[u] = base_adj[u]
        
        # Add copies and edges
        # For each edge (u, v) in base:
        # Add (u, v_copy) and (v, u_copy)
        for u in range(base_n):
            mask = base_adj[u]
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u: # Avoid double processing
                    # u connected to v_copy
                    adj[u] |= (1 << (base_n + v))
                    adj[base_n + v] |= (1 << u)
                    # v connected to u_copy
                    adj[v] |= (1 << (base_n + u))
                    adj[base_n + u] |= (1 << v)
                m ^= lsb
        
        # Connect w to all copies
        w = base_n + base_n
        for i in range(base_n):
            adj[w] |= (1 << (base_n + i))
            adj[base_n + i] |= (1 << w)
            
        return adj

    def make_c5_adj():
        adj = [0] * 5
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def make_c7_adj():
        adj = [0] * 7
        for i in range(7):
            adj[i] |= (1 << ((i+1)%7))
            adj[i] |= (1 << ((i-1)%7))
        return adj

    def make_c5_plus_isolated_adj(k):
        # C5 on 0..4, rest isolated
        adj = [0] * k
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def greedy_expand(adj):
        # Randomized greedy expansion
        # We try to add edges in random order repeatedly
        # until no more edges can be added
        
        # List of all possible edges
        possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        
        # We don't know how many passes we need, so we loop until stable
        # But to allow diversity, we can limit passes or just run until stable
        # Running until stable is safe and fast for N=16
        
        changed = True
        while changed:
            changed = False
            # Shuffle the possible list? 
            # If we shuffle the original list, we might miss edges that become possible later?
            # No, if an edge is not possible now, it might become possible later?
            # Adding edges only REMOVES possibilities (creates triangles).
            # So if an edge is not possible now, it will never be possible later.
            # Wait, is that true?
            # Triangle-free: adding edge (u,v) requires N(u) and N(v) to be disjoint.
            # If we add other edges, N(u) and N(v) grow.
            # So if they are not disjoint now, they won't be disjoint later.
            # Therefore, the set of addable edges only SHRINKS.
            # So we just need to iterate through the initial set of possible edges.
            # If we add an edge, we don't need to re-check edges that were already rejected?
            # Yes, if (u,v) was rejected because of a common neighbor w, adding more edges won't remove w.
            # So we can just iterate through the initial list of possible edges once?
            # No, the order matters.
            # If we try (u,v) and reject it, it stays rejected.
            # So we can just shuffle the list of all non-existent edges and try them in that order.
            # If we add an edge, it might block others, but we've already decided on the order.
            # This is a standard randomized greedy.
            
            random.shuffle(possible)
            for u, v in possible:
                # Re-check if it's still not present and triangle-free
                # It's not present because we only add.
                if is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
                    # Note: We don't remove from 'possible' list, but the check handles it.
                    # Actually, we can just break if no more can be added?
                    # No, we continue to try others.
        
        return adj

    def generate_seed():
        r = random.random()
        if r < 0.3:
            # Grötzsch (Mycielski C5)
            base = make_c5_adj()
            return make_mycielski(base, 5, n)
        elif r < 0.6:
            # Mycielski C7
            base = make_c7_adj()
            return make_mycielski(base, 7, n)
        else:
            # Mycielski C5 + isolated
            # k=6 -> 13 vertices
            # k=7 -> 15 vertices
            if random.random() < 0.5:
                base = make_c5_plus_isolated_adj(6)
            else:
                base = make_c5_plus_isolated_adj(7)
            return make_mycielski(base, len(base), n)

    while time.time() < deadline:
        adj = generate_seed()
        greedy_expand(adj)
        edges = get_edges_from_adj(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    return best_edges
