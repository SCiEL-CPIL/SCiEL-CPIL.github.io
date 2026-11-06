import random

def construct():
    n = 20
    best_edges = []
    best_count = 0
    
    def get_color(edges, u, v):
        key = (min(u, v), max(u, v))
        return edges.get(key)
        
    def is_rainbow(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) == 4
        
    def try_repair(seed_val):
        random.seed(seed_val)
        edges = {}
        adj = [set() for _ in range(n)]
        
        for u in range(n):
            for v in range(u+1, n):
                c = random.randint(0, 19)
                edges[(u, v)] = c
                adj[u].add(v)
                adj[v].add(u)
                
        max_iter = 2000
        for _ in range(max_iter):
            bad_cycle = None
            # Find a bad cycle
            # Optimization: Only check if edge exists
            for a in range(n):
                for b in range(a+1, n):
                    if b not in adj[a]: continue
                    for c in range(b+1, n):
                        if c not in adj[b]: continue
                        for d in range(c+1, n):
                            # Check cycle a-b-c-d-a
                            if d in adj[c] and a in adj[d]:
                                c1 = get_color(edges, a, b)
                                c2 = get_color(edges, b, c)
                                c3 = get_color(edges, c, d)
                                c4 = get_color(edges, d, a)
                                if not is_rainbow(c1, c2, c3, c4):
                                    bad_cycle = (a, b, c, d, [(a,b), (b,c), (c,d), (d,a)])
                                    break
                        if bad_cycle: break
                    if bad_cycle: break
                if bad_cycle: break
            if bad_cycle: break
            
            if bad_cycle is None:
                break
                
            a, b, c, d, cycle_edges = bad_cycle
            
            fixed = False
            # Shuffle cycle edges to try different ones first
            random.shuffle(cycle_edges)
            for u, v in cycle_edges:
                old_c = get_color(edges, u, v)
                for new_c in range(20):
                    if new_c == old_c: continue
                    key = (min(u,v), max(u,v))
                    edges[key] = new_c
                    
                    valid = True
                    # Check 4-cycles containing (u,v)
                    for x in adj[u]:
                        if x == v: continue
                        for y in adj[v]:
                            if y == u or y == x: continue
                            if y not in adj[x]: continue
                            c_ux = get_color(edges, u, x)
                            c_xy = get_color(edges, x, y)
                            c_yv = get_color(edges, y, v)
                            c_vu = new_c
                            if not is_rainbow(c_ux, c_xy, c_yv, c_vu):
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        fixed = True
                        break
                    else:
                        edges[key] = old_c
                if fixed:
                    break
                    
            if not fixed:
                # Remove edge
                u, v = cycle_edges[0]
                key = (min(u, v), max(u, v))
                del edges[key]
                adj[u].remove(v)
                adj[v].remove(u)
                
        return len(edges), edges

    # Run multiple attempts
    for s in range(20):
        count, edges = try_repair(s)
        if count > best_count:
            best_count = count
            best_edges = edges
            
    result = []
    for (u, v), c in best_edges.items():
        result.append([u, v, c])
        
    return result
