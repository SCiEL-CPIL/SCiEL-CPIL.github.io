<<<<<<< SEARCH
def construct():
    deadline = time.time() + 120  # Safety margin before 150s limit
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
        
        if time.time() > deadline - 5:
            # Not enough time for remaining instances; use trivial decomposition
            for n2, p2, seed2 in specs[len(results):]:
                edges2 = _instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
=======
def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
        
        if time.time() > deadline - 5:
            # Not enough time for remaining instances; use trivial decomposition
            for n2, p2, seed2 in specs[len(results):]:
                edges2 = _instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
>>>>>>> REPLACE
