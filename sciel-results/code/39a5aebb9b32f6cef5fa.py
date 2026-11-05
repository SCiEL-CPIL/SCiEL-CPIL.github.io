def construct():
    import time
    start_time = time.time()
    deadline = start_time + 115
    
    # M12 generators (0-indexed), perm[i] = image of i
    g1 = tuple((i + 1) % 12 for i in range(12))
    g2 = (3, 11, 5, 9, 10, 8, 7, 1, 0, 4, 6, 2)
    
    def invert(p):
        inv = [0] * 12
        for i in range(12):
            inv[p[i]] = i
        return tuple(inv)
    
    g1_inv = invert(g1)
    g2_inv = invert(g2)
    generators = (g1, g2, g1_inv, g2_inv)
    
    identity = tuple(range(12))
    
    def compose(p, q):
        return tuple(p[q[i]] for i in range(12))
    
    base_block = frozenset([0, 1, 2, 3, 4, 5])
    
    seen_groups = {identity}
    queue = [identity]
    blocks = {base_block}
    
    idx = 0
    while idx < len(queue):
        if time.time() > deadline:
            break
        g = queue[idx]
        idx += 1
        for gen in generators:
            h = compose(gen, g)
            if h not in seen_groups:
                seen_groups.add(h)
                queue.append(h)
                b = frozenset(h[i] for i in base_block)
                blocks.add(b)
    
    return [list(b) for b in blocks]
