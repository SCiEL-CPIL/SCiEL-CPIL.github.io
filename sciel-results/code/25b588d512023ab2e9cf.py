import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * 15 + (v - u - 1)
    
    # Precompute Q3 patterns: for each pair of adjacent vertices in Q3,
    # store the required structure as (neighbor_pairs, common_pairs)
    # Q3 vertices: 0-7, edges: (0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)
    # For edge (u,v) in graph, u,v map to two adjacent Q3 vertices.
    # By symmetry, WLOG u->0, v->1
    # Then: a,b from N(0)\{1} = {2,3}; c,d from N(1)\{0} = {4,5}
    # a-c or a-d, b-d or b-c must exist (matching)
    # e in N(a)∩N(b)\{u,v,a,b} (maps to 6)
    # f in N(c)∩N(d)\{u,v,c,d} (maps to 7)  
    # e-f must exist
    
    def has_q3_with_edge(u, v, adj):
        """Fast check if adding edge (u,v) creates a Q3 containing that edge."""
        mask_uv = (1 << u) | (1 << v)
        neu = adj[u] & ~mask_uv
        nev = adj[v] & ~mask_uv
        
        if neu.bit_count() < 2 or nev.bit_count() < 2:
            return False
        
        # Get lists
        neu_list = [i for i in range(n) if (neu >> i) & 1]
        nev_list = [i for i in range(n) if (nev >> i) & 1]
        
        for a, b in itertools.combinations(neu_list, 2):
            common_ab = adj[a] & adj[b] & ~((1<<a)|(1<<b)|mask_uv)
            if common_ab == 0:
                continue
            for c, d in itertools.combinations(nev_list, 2):
                if c in (a,b) or d in (a,b):
                    continue
                # Check matching: a-c & b-d OR a-d & b-c
                ac_bd = (adj[a] & (1<<c)) and (adj[b] & (1<<d))
                ad_bc = (adj[a] & (1<<d)) and (adj[b] & (1<<c))
                if not (ac_bd or ad_bc):
                    continue
                common_cd = adj[c] & adj[d] & ~((1<<c)|(1<<d)|mask_uv|(1<<a)|(1<<b))
                if common_cd == 0:
                    continue
                e_mask = common_ab & ~((1<<c)|(1<<d))
                f_mask = common_cd & ~((1<<a)|(1<<b))
                # Check if any e in e_mask is adjacent to any f in f_mask
                e_bits = e_mask
                while e_bits:
                    lb = e_bits & (-e_bits)
                    e_idx = lb.bit_length() - 1
                    if adj[e_idx] & f_mask:
                        return True
                    e_bits ^= lb
        return False
    
    # Greedy initialization: add edges in random order if they don't create Q3
    adj = [0] * n
    random.seed(42)
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if has_q3_with_edge(u, v, adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    current_edges = sum(adj[i].bit_count() for i in range(n)) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # Simulated Annealing
    def get_missing(adj):
        return [(i,j) for i in range(n) for j in range(i+1,n) if not (adj[i] & (1<<j))]
    
    def get_existing(adj):
        return [(i,j) for i in range(n) for j in range(i+1,n) if adj[i] & (1<<j)]
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Try to add a random missing edge
        missing = get_missing(adj)
        if missing:
            u, v = random.choice(missing)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not has_q3_with_edge(u, v, adj):
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Try edge swap: remove one existing, add one missing
        existing = get_existing(adj)
        missing = get_missing(adj)
        if existing and missing:
            u1, v1 = random.choice(existing)
            u2, v2 = random.choice(missing)
            
            # Remove (u1,v1)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            current_edges -= 1
            
            # Add (u2,v2)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            if not has_q3_with_edge(u2, v2, adj):
                current_edges += 1
                delta = 0  # same number of edges
                # Accept if better or with probability based on T
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Revert the add, keep the remove (or revert both)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                # Decide whether to keep the removal
                # This is a "kick" - accept with probability
                if random.random() > T:
                    # Revert removal too
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    current_edges += 1
                else:
                    # Keep removal, try to add another edge
                    missing2 = get_missing(adj)
                    if missing2:
                        u3, v3 = random.choice(missing2)
                        adj[u3] |= (1 << v3)
                        adj[v3] |= (1 << u3)
                        if not has_q3_with_edge(u3, v3, adj):
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                        else:
                            adj[u3] &= ~(1 << v3)
                            adj[v3] &= ~(1 << u3)
        
        T *= cooling
    
    # Final polish: try all possible single edge additions
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not (adj[u] & (1 << v)):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not has_q3_with_edge(u, v, adj):
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    return get_edges_from_adj(best_adj)
