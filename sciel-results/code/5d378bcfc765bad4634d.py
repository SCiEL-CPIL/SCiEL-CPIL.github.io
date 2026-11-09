def _search(target_n, base_edges, deadline):
    random.seed(42)
    for it in range(100000):
        if time.time() > deadline - 5:
            break
        
        edges = set(base_edges)
        
        # Randomly remove 0-3 edges from base
        base_list = list(base_edges)
        num_remove = random.randint(0, 3)
        for _ in range(num_remove):
            e = random.choice(base_list)
            edges.discard(e)
        
        # Add new vertices
        for new_v in range(32, target_n):
            existing = list(range(new_v))
            # Connect to a random subset
            num_conn = random.randint(4, min(12, len(existing)))
            neighbors = random.sample(existing, num_conn)
            for nb in neighbors:
                edges.add(frozenset((new_v, nb)))
        
        if _has_c5(target_n, edges):
            continue
        if _has_independent_set(target_n, edges, 9):
            continue
        
        return edges
    
    return None
