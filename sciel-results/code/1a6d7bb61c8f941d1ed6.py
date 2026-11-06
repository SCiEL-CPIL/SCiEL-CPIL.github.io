import random
from itertools import combinations

def construct():
    n = 16
    # Build 1-factorisation of K16
    inf = 15
    mod = 15
    matchings = []
    for k in range(mod):
        m = [(inf, k)]
        for i in range(1, 8):
            a = (k + i) % mod
            b = (k - i) % mod
            m.append((a, b))
        matchings.append(m)
    
    colour = {}
    for c, m in enumerate(matchings):
        for (u, v) in m:
            colour[(u, v)] = c
            colour[(v, u)] = c
    
    all_k4s = list(combinations(range(n), 4))
    
    def count_distinct_in_k4(quad, colour_map):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        cols = set()
        for (u, v) in edges:
            cols.add(colour_map[(u, v)])
        return len(cols)
    
    # Precompute K4s containing each edge
    edge_to_k4s = {}
    for idx, quad in enumerate(all_k4s):
        a, b, c, d = quad
        edges_in_quad = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        for (u, v) in edges_in_quad:
            key = (min(u,v), max(u,v))
            if key not in edge_to_k4s:
                edge_to_k4s[key] = []
            edge_to_k4s[key].append(idx)
    
    random.seed(42)
    best_colour = dict(colour)
    best_num = len(set(best_colour.values()))
    
    for iteration in range(3000):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        old_col = best_colour[(u, v)]
        existing_cols = sorted(set(best_colour.values()))
        random.shuffle(existing_cols)
        
        for new_col in existing_cols:
            if new_col == old_col:
                continue
            
            best_colour[(u, v)] = new_col
            
            ok = True
            for k4_idx in edge_to_k4s.get((u, v), []):
                quad = all_k4s[k4_idx]
                if count_distinct_in_k4(quad, best_colour) < 5:
                    ok = False
                    break
            
            if ok:
                new_num = len(set(best_colour.values()))
                if new_num < best_num:
                    best_num = new_num
                    break
                else:
                    best_colour[(u, v)] = old_col
            else:
                best_colour[(u, v)] = old_col
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(best_colour[(i, j)])
    return result
