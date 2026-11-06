import time
import random

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        color = [-1] * n
        
        def backtrack():
            best_v = -1
            best_avail = 7
            for i in range(n):
                if color[i] != -1:
                    continue
                mask = adj[i]
                used = 0
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if color[j] != -1:
                        used |= (1 << color[j])
                    m ^= lsb
                avail = 7 & ~used
                if avail == 0:
                    return False
                if bin(avail).count('1') < bin(best_avail).count('1'):
                    best_avail = avail
                    best_v = i
                    if bin(avail).count('1') == 1:
                        break
            
            if best_v == -1:
                return True
            
            m = best_avail
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = -1
                m ^= lsb
            return False
        
        return backtrack()

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

    def make_grotzsch_adj():
        adj = [0] * n
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
        return adj

    def greedy_expand(adj, num_passes=5):
        for _ in range(num_passes):
            possible = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        possible.append((u, v))
            random.shuffle(possible)
            for u, v in possible:
                if is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def local_search_polish(adj):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        continue
                    if is_triangle_free_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        return adj

    while time.time() < deadline:
        r = random.random()
        if r < 0.6:
            adj = make_grotzsch_adj()
        else:
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.55:
                        if is_triangle_free_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        
        adj = greedy_expand(adj, num_passes=5)
        adj = local_search_polish(adj)
        
        if not is_3_colorable(adj):
            edges = get_edges_from_adj(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    return best_edges
