import time
import random

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        attempts = 20 if n > 45 else 10
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            result = _solve_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                return result
                
    return {'n': 1, 'edges': []}

def _solve_c4_free(n, min_deg_req, deadline, seed_offset):
    rng = random.Random(seed_offset * 7919 + 31)
    adj_mask = [0] * n
    two_step = [0] * n
    
    def popcount(x):
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        return (x * 0x0101010101010101) >> 56

    def get_degree(u):
        return popcount(adj_mask[u])
    
    def rebuild_two_step():
        for i in range(n):
            two_step[i] = 0
            m = adj_mask[i]
            while m:
                lsb = m & -m
                a = lsb.bit_length() - 1
                two_step[i] |= adj_mask[a]
                m -= lsb
            two_step[i] &= ~(1 << i)
    
    def can_add_edge(u, v):
        if (adj_mask[u] >> v) & 1:
            return False
        return (two_step[u] & adj_mask[v]) == 0

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        rebuild_two_step()

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        rebuild_two_step()

    # Phase 1: Greedy
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if can_add_edge(u, v):
            add_edge(u, v)
    
    # Phase 2: Local Search
    max_iters = 50000
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
            
        degrees = [get_degree(u) for u in range(n)]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                m = adj_mask[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    m -= lsb
            return {'n': n, 'edges': edges}
        
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        candidates = [u for u in range(n) if u != v and not ((adj_mask[v] >> u) & 1)]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if can_add_edge(u, v):
                add_edge(u, v)
                added = True
                break
        
        if not added:
            non_neighbors = [u for u in range(n) if u != v and not ((adj_mask[v] >> u) & 1)]
            blocking_edges = set()
            for u in non_neighbors[:3]:
                common = two_step[u] & adj_mask[v]
                if common:
                    m = common
                    while m:
                        lsb = m & -m
                        b = lsb.bit_length() - 1
                        ma = adj_mask[u]
                        while ma:
                            lsb2 = ma & -ma
                            a = lsb2.bit_length() - 1
                            if (adj_mask[a] >> b) & 1:
                                blocking_edges.add((min(a,b), max(a,b)))
                                break
                            ma -= lsb2
                        m -= lsb
            
            if blocking_edges:
                edge = rng.choice(list(blocking_edges))
                remove_edge(edge[0], edge[1])
            elif any(adj_mask[i] for i in range(n)):
                while True:
                    i = rng.randint(0, n-1)
                    if adj_mask[i]:
                        m = adj_mask[i]
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        remove_edge(i, j)
                        break
            else:
                break
    
    return None
