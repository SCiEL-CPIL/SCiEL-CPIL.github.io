    for seed in range(30):
        if time.time() > deadline:
            break
        random.seed(seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        family = greedy_with_ordering(indices)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    if best_family and time.time() < deadline - 5:
        improved_family = local_search(best_family[:], max_iterations=50)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family
            
    # Maybe one more pass
    if time.time() < deadline - 2:
         improved_family = local_search(best_family[:], max_iterations=50)
         if len(improved_family) > best_size:
             best_size = len(improved_family)
             best_family = improved_family
