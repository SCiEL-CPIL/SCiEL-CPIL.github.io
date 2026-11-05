def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 80.0
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    n = 128
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a - 1, b - 1, c - 1))
    
    num_triples = len(bad_triples)
    
    elem_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    active = [True] * n
    triple_active = [True] * num_triples
    triple_count = [3] * num_triples
    elem_count = [len(elem_triples[i]) for i in range(n)]
    
    active_triple_set = set(range(num_triples))
    
    while active_triple_set:
        if time.time() > deadline - 5.0:
            break
        
        best_t = -1
        best_sum = -1
        for t in active_triple_set:
            a, b, c = bad_triples[t]
            s = elem_count[a] + elem_count[b] + elem_count[c]
            if s > best_sum:
                best_sum = s
                best_t = t
        
        if best_t < 0:
            break
        
        a, b, c = bad_triples[best_t]
        candidates = [(elem_count[a], a), (elem_count[b], b), (elem_count[c], c)]
        candidates.sort(reverse=True)
        rem = candidates[0][1]
        
        active[rem] = False
        
        for t in elem_triples[rem]:
            if not triple_active[t]:
                continue
            triple_count[t] -= 1
            if triple_count[t] == 0:
                triple_active[t] = False
                active_triple_set.discard(t)
            else:
                ta, tb, tc = bad_triples[t]
                for other in (ta, tb, tc):
                    if other != rem and active[other]:
                        elem_count[other] -= 1
        
        elem_count[rem] = 0
    
    result = [i + 1 for i in range(n) if active[i]]
    if not result:
        result = [1]
    return result
