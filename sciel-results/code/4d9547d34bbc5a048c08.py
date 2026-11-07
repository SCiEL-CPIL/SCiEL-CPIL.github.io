<<<<<<< SEARCH
    # Try random subsets of various sizes
    random.seed(42)
    for k in range(18, 20):
        for _ in range(30):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), k))
            try_vertex_subset(subset, deadline)
    
    # Try subsets of size 20 with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(10):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), 20))
            try_vertex_subset(subset, deadline)
=======
    # Try random subsets of various sizes
    random.seed(42)
    for k in range(16, 20):
        for _ in range(50):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), k))
            try_vertex_subset(subset, deadline)
    
    # Try all pairs of vertex removals
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            subset = all_vertices - {i, j}
            try_vertex_subset(subset, deadline)
    
    # Try subsets of size 19 with different random seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(15):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), 19))
            try_vertex_subset(subset, deadline)
>>>>>>> REPLACE
