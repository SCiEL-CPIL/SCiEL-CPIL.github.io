def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    all_blocks, Cover, n_blocks, n_tsubs = _prepare(v, k, t)

    best_selected = None
    best_count = float('inf')

    # Structural Change: Remove slow local search, increase greedy restarts
    # Greedy is ~10s per run. 145s / 10s ~ 14 restarts.
    # We will use a wider range of seeds.
    
    # Reserve time for final check
    restart_budget = deadline - 5.0

    # Generate many seeds
    base_seeds = [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 10101, 20202, 30303, 40404, 50505]
    
    for seed in base_seeds:
        if time.time() > restart_budget:
            break
        rng = random.Random(seed)
        
        # Greedy construction
        selected, uncovered = _greedy_construct(Cover, n_tsubs, deadline - 10.0, rng)
        
        if np.any(uncovered):
            continue

        # Remove redundant blocks
        selected = _remove_redundant(selected, Cover, n_tsubs, deadline - 5.0)
        
        count = len(selected)
        if count < best_count:
            best_count = count
            best_selected = list(selected)
        
        # Early termination if we hit a very good solution
        if best_count <= 140:
            break

    if best_selected is None:
        # Fallback: simple greedy
        rng = random.Random(12345)
        best_selected, _ = _greedy_construct(Cover, n_tsubs, deadline - 5.0, rng)
        best_selected = _remove_redundant(best_selected, Cover, n_tsubs, deadline - 2.0)

    # Convert to list of blocks
    result = [list(all_blocks[i]) for i in best_selected]
    return result
