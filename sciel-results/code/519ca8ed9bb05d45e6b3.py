def greedy_initial_mask_smart(seed_val=0):
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    
    # Add small random noise to break ties and diversify
    noise = {x: random.random() for x in candidates}
    candidates.sort(key=lambda x: -(max_gain_3aps[x] + noise[x]))
    
    for x in candidates:
        if is_4ap_free_addition_mask(A_mask, x):
            A_mask |= (1 << (x - 1))
    return A_mask
