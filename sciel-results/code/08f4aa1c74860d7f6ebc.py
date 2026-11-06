import time
import sys

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
                
    # Build conflict graph (adjacency bitmasks)
    # Two edges conflict if they share a C4 face.
    adj = [0] * n_edges
    
    for i_dim in range(d):
        for j_dim in range(i_dim + 1, d):
            others = [k for k in range(d) if k != i_dim and k != j_dim]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i_dim)
                v3 = u ^ (1 << j_dim)
                v4 = u ^ (1 << i_dim) ^ (1 << j_dim)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                # Connect all edges in this face
                for e_a in [e1, e2, e3, e4]:
                    for e_b in [e1, e2, e3, e4]:
                        if e_a != e_b:
                            adj[e_a] |= (1 << e_b)
                            
    # B&B Solver for Maximum Independent Set
    best_size = 0
    best_mask = 0
    start_time = time.time()
    timeout = 115.0
    
    sys.setrecursionlimit(1000)
    
    def greedy_is(cands):
        is_mask = 0
        temp = cands
        while temp:
            # Pick vertex with highest degree in temp
            max_deg = -1
            max_v = -1
            t = temp
            while t:
                v = (t & -t).bit_length() - 1
                deg = (adj[v] & temp).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    max_v = v
                t &= ~(1 << v)
            
            if max_v == -1:
                break
                
            is_mask |= (1 << max_v)
            temp &= ~adj[max_v]
            temp &= ~(1 << max_v)
        return is_mask

    # Initial solution
    init_is = greedy_is((1 << n_edges) - 1)
    best_size = init_is.bit_count()
    best_mask = init_is
    
    def dfs(cands, curr_mask, curr_size):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > timeout:
            return
            
        # Upper bound: current size + number of candidates
        if curr_size + cands.bit_count() <= best_size:
            return
            
        if cands == 0:
            if curr_size > best_size:
                best_size = curr_size
                best_mask = curr_mask
            return
            
        # Heuristic: Pick vertex with max degree in subgraph
        max_deg = -1
        max_v = -1
        t = cands
        while t:
            v = (t & -t).bit_length() - 1
            deg = (adj[v] & cands).bit_count()
            if deg > max_deg:
                max_deg = deg
                max_v = v
            t &= ~(1 << v)
            
        if max_v == -1:
            return
            
        # Branch 1: Include max_v
        new_cands = cands & ~adj[max_v] & ~(1 << max_v)
        dfs(new_cands, curr_mask | (1 << max_v), curr_size + 1)
        
        # Branch 2: Exclude max_v
        new_cands2 = cands & ~(1 << max_v)
        dfs(new_cands2, curr_mask, curr_size)

    dfs((1 << n_edges) - 1, 0, 0)
    
    # Construct result
    result = []
    for i in range(n_edges):
        if best_mask & (1 << i):
            result.append(list(edges[i]))
            
    return result
