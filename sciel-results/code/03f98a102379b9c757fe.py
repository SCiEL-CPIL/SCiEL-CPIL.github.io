def _generate_random_linear_graph():
    triples = []
    used_pairs = set()
    
    # All possible triples
    all_triples = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                all_triples.append((i, j, k))
                
    random.shuffle(all_triples)
    
    for t in all_triples:
        a, b, c = t
        p1 = (a, b)
        p2 = (a, c)
        p3 = (b, c)
        
        if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
            continue
            
        triples.append([a, b, c])
        used_pairs.add(p1)
        used_pairs.add(p2)
        used_pairs.add(p3)
        
    return triples
