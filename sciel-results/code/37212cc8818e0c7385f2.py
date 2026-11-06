def construct():
    N = 180
    triples = []
    triples_by_node = {i: [] for i in range(1, N + 1)}
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = 2 * b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    t = (a, b, c)
                    triples.append(t)
                    triples_by_node[a].append(t)
                    triples_by_node[b].append(t)
                    triples_by_node[c].append(t)
                    
    S = set(range(1, N + 1))
    deg = {i: len(triples_by_node[i]) for i in S}
    removed = []
    
    while True:
        active_triple = None
        for t in triples:
            if t[0] in S and t[1] in S and t[2] in S:
                active_triple = t
                break
                
        if not active_triple:
            break
            
        a, b, c = active_triple
        candidates = [a, b, c]
        candidates.sort(key=lambda x: deg[x], reverse=True)
        v_to_remove = candidates[0]
        
        S.remove(v_to_remove)
        removed.append(v_to_remove)
        
        for t in triples_by_node[v_to_remove]:
            nodes = [x for x in t if x != v_to_remove]
            n1, n2 = nodes
            if n1 in S and n2 in S:
                deg[n1] -= 1
                deg[n2] -= 1
                
        deg[v_to_remove] = 0
        
    # Try to add back
    # Sort removed by some heuristic, e.g. original degree or just value
    # Let's try adding back in order of removal (reverse) or by value
    for v in reversed(removed):
        # Check if v can be added
        can_add = True
        for t in triples_by_node[v]:
            nodes = [x for x in t if x != v]
            n1, n2 = nodes
            if n1 in S and n2 in S:
                can_add = False
                break
        if can_add:
            S.add(v)
            
    return list(S)
