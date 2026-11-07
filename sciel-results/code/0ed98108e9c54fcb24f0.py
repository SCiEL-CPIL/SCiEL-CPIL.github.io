def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = P['n']
    min_v = P['minimum_vertices']

    best_cnt = 0
    best_vertices = None

    count = 0
    for k in range(min_v, n + 1):
        for comb in itertools.combinations(range(n), k):
            count += 1
            if count % 1000 == 0 and time.time() > deadline:
                if best_vertices:
                    return best_vertices
                else:
                    return list(range(min_v))
            
            mask = 0
            for v in comb:
                mask |= 1 << v
            
            degs = set()
            for v in comb:
                degs.add(bin(mask & _ADJ_MASKS[v]).count('1'))
            
            cnt = len(degs)
            if cnt > best_cnt:
                best_cnt = cnt
                best_vertices = list(comb)
                if best_cnt == n:
                    return best_vertices

    if best_vertices is None:
        return list(range(min_v))
        
    return best_vertices
