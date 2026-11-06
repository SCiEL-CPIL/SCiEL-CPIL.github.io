def construct():
    # Generators for the Mathieu group M12
    # a: (0 1 2 3 4 5 6 7 8 9 10)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    # b: (1 3 4 5)(2 8 10 6)
    b = [0, 3, 8, 4, 5, 1, 2, 7, 10, 9, 6, 11]
    # c: (0 11)(1 2)(3 4)(5 6)(7 8)(9 10)
    c = [11, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 0]
    
    def compose(p1, p2):
        # (p1 o p2)(x) = p1(p2(x))
        return [p1[p2[i]] for i in range(12)]
        
    group = set()
    queue = []
    identity = tuple(range(12))
    group.add(identity)
    queue.append(identity)
    
    generators = [tuple(a), tuple(b), tuple(c)]
    
    while queue:
        g = queue.pop(0)
        for gen in generators:
            new_g = compose(gen, g)
            if new_g not in group:
                group.add(new_g)
                queue.append(new_g)
                
            new_g2 = compose(g, gen)
            if new_g2 not in group:
                group.add(new_g2)
                queue.append(new_g2)
                
    seed_block = frozenset([0, 1, 2, 3, 4, 5])
    blocks = set()
    
    for g in group:
        mapped_block = frozenset([g[i] for i in seed_block])
        blocks.add(mapped_block)
        
    result = [list(b) for b in blocks]
    return result
