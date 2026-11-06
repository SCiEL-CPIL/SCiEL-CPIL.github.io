import random

def construct():
    n = 20
    vertices = list(range(n))
    
    def try_greedy_structured(seed_val):
        edge_colors = {}
        adj = [set() for _ in range(n)]
        
        def add_edge(u, v, c):
            key = (min(u, v), max(u, v))
            edge_colors[key] = c
            adj[u].add(v)
            adj[v].add(u)

        def is_valid(u, v, c):
            # Check all 4-cycles u-a-b-v
            for a in adj[u]:
                if a == v: continue
                c_ua = edge_colors.get((min(u, a), max(u, a)))
                if c_ua is None: continue
                for b in adj[v]:
                    if b == u or b == a: continue
                    key_ab = (min(a, b), max(a, b))
                    c_ab = edge_colors.get(key_ab)
                    if c_ab is None: continue
                    c_bv = edge_colors.get((min(b, v), max(b, v)))
                    # Check distinctness
                    if c == c_ua or c == c_ab or c == c_bv:
                        return False
                    if c_ua == c_ab or c_ua == c_bv or c_ab == c_bv:
                        return False
            return True

        # Phase 1: Bipartite K(10,10)
        a_part = list(range(10))
        b_part = list(range(10, 20))
        bip_edges = [(u, v) for u in a_part for v in b_part]
        
        rng = random.Random(seed_val)
        rng.shuffle(bip_edges)
        
        for u, v in bip_edges:
            key = (min(u, v), max(u, v))
            if key in edge_colors: continue
            for c in range(20):
                if is_valid(u, v, c):
                    add_edge(u, v, c)
                    break

        # Phase 2: Intra-part edges
        intra_edges = [(u, v) for u in a_part for v in a_part if u < v]
        intra_edges += [(u, v) for u in b_part for v in b_part if u < v]
        rng.shuffle(intra_edges)
        
        for u, v in intra_edges:
            key = (min(u, v), max(u, v))
            if key in edge_colors: continue
            for c in range(20):
                if is_valid(u, v, c):
                    add_edge(u, v, c)
                    break

        return len(edge_colors), edge_colors

    best_count = 0
    best_list = []
    
    # Run multiple seeds
    for i in range(30):
        count, edict = try_greedy_structured(i)
        if count > best_count:
            best_count = count
            best_list = [[u, v, c] for (u, v), c in edict.items()]
            
    return best_list
