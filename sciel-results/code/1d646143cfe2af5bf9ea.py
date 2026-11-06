def construct():
    N = 180
    # Pre-compute all reciprocal triples {a, b, c} with a < b < c
    triples = []
    for a in range(1, N + 1):
        max_b = min(N, 2 * a - 1)
        for b in range(a + 1, max_b + 1):
            denom = b - a
            if (a * b) % denom == 0:
                c = (a * b) // denom
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # Map element to list of triple indices it belongs to
    element_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        element_triples[a].append(idx)
        element_triples[b].append(idx)
        element_triples[c].append(idx)
        
    # Calculate degree of each element (number of triples it participates in)
    degrees = {i: len(element_triples[i]) for i in range(1, N + 1)}
    
    # Sort elements by degree (ascending)
    # Elements with fewer conflicts are added first
    sorted_elements = sorted(range(1, N + 1), key=lambda x: (degrees[x], x))
    
    S = set()
    
    # Greedy addition: add element if it doesn't complete any forbidden triple
    for x in sorted_elements:
        is_safe = True
        for t_idx in element_triples[x]:
            a, b, c = triples[t_idx]
            # Identify the other two elements in this triple
            if x == a:
                others = (b, c)
            elif x == b:
                others = (a, c)
            else:
                others = (a, b)
            
            # If both other elements are already in S, adding x would form a forbidden triple
            if others[0] in S and others[1] in S:
                is_safe = False
                break
        
        if is_safe:
            S.add(x)
            
    return list(S)
