def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill
    
    n = 12
    min_deg = 3
    min_girth = 4
    
    best_score = float('inf')
    best_edges = None
    
    # Strategy 1: Try known structured constructions
    constructions = []
    
    # Prism graph (current seed)
    es = set()
    for i in range(6):
        for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
            es.add(tuple(sorted((u, v))))
    constructions.append(es)
    
    # Two K_{3,3}
    es2 = set()
    for i in range(3):
        for j in range(3, 6):
            es2.add((i, j))
    for i in range(6, 9):
        for j in range(9, 12):
            es2.add((i, j))
    constructions.append(es2)
            
    # Twisted prism
    es3 = set()
    for i in range(6):
        es3.add((i, (i+1)%6))
        es3.add((i+6, (i+1)%6+6))
    cross = [(0,6),(1,8),(2,10),(3,11),(4,7),(5,9)]
    for u,v in cross:
        es3.add((u,v))
    constructions.append(es3)
    
    # Evaluate all constructions
    for es in constructions:
        edges = _make_edges(es)
        valid, ncl = _evaluate(edges)
        if valid and ncl < best_score:
            best_score = ncl
            best_edges = edges[:]
            if best_score == 1:
                return best_edges

    # ... rest of random search ...
