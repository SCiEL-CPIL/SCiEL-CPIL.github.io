def construct(k, g):
    q = k - 1
    d = (g + 1) // 2
    if 2 * d + 1 < g:
        d += 1
        
    vertices = []
    for i in range(q ** d):
        s = [(i // (q ** j)) % q for j in range(d)]
        valid = True
        for j in range(d - 1):
            if s[j] == s[j + 1]:
                valid = False
                break
        if valid:
            vertices.append(tuple(s))
            
    n = len(vertices)
    index = {v: i for i, v in enumerate(vertices)}
    edges = set()
    
    for v in vertices:
        i = index[v]
        # Type 1: u[1:] == v[:-1]
        prefix = v[:-1]
        # u = (u0, prefix[0], ..., prefix[-1])
        # u[-1] = prefix[-1] = v[-2]
        # Condition: u[-1] != v[0] => v[-2] != v[0]
        if v[-2] != v[0]:
            for u0 in range(q):
                if u0 != v[0]: # u must be valid
                    u = (u0,) + prefix
                    if u in index:
                        edges.add((i, index[u]) if i < index[u] else (index[u], i))
                        
        # Type 2: w[:-1] == v[1:]
        suffix = v[1:]
        # w = (w0, suffix[0], ..., suffix[-1])
        # w[0] != v[-1]
        for w0 in range(q):
            if w0 != v[-1]:
                w = (w0,) + suffix
                if w in index:
                    edges.add((i, index[w]) if i < index[w] else (index[w], i))
                    
    return n, list(edges)
