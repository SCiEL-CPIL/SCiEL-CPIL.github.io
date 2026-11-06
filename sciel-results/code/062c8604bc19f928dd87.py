import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    n = 16
    
    # Precompute subsets for each edge (u, v)
    # subsets_for_edge[(u, v)] = list of 8-tuples
    subsets_for_edge = {}
    for u in range(n):
        for v in range(u + 1, n):
            remaining = [x for x in range(n) if x != u and x != v]
            subs = []
            for s in itertools.combinations(remaining, 6):
                subs.append(tuple(list(s) + [u, v]))
            subsets_for_edge[(u, v)] = subs

    def is_q3(vertices, adj):
        # vertices is a tuple of 8 ints
        v_list = list(vertices)
        deg = [0] * 8
        edge_count = 0
        
        for i in range(8):
            u = v_list[i]
            for j in range(i+1, 8):
                v = v_list[j]
                if (adj[u] >> v) & 1:
                    edge_count += 1
                    deg[i] += 1
                    deg[j] += 1
        
        if edge_count != 12:
            return False
            
        for d in deg:
            if d != 3:
                return False
                
        # Check bipartite
        local_adj = [0] * 8
        for i in range(8):
            u = v_list[i]
            mask = adj[u]
            for j in range(8):
                v = v_list[j]
                if (mask >> v) & 1:
                    local_adj[i] |= (1 << j)
                    
        colors = [-1] * 8
        for start in range(8):
            if colors[start] != -1:
                continue
            colors[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                mask = local_adj[u]
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        stack.append(v)
                    elif colors[v] == colors[u]:
                        return False
                    m ^= lsb
        return True

    def try_add_edge(u, v, adj):
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check for Q3 using precomputed subsets
        for verts in subsets_for_edge[(u, v)]:
            if is_q3(verts, adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                return False
        return True

    best_edges = 0
    best_adj = [0] * n
    
    while time.time() - start_time < time_limit:
        adj = [0] * n
        vertices = list(range(n))
        random.shuffle(vertices)
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if ((adj[u] >> v) & 1):
                continue
            try_add_edge(u, v, adj)
            
        edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
