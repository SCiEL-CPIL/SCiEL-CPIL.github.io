def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Option 1: Odd-Even Transposition
    seed1 = make_seed()
    # Option 2: Greedy Construction
    seed2 = greedy_construct(rng)
    
    # Pick the better seed (shorter after greedy removal)
    seed1_removed = greedy_remove(seed1, rng)
    seed2_removed = greedy_remove(seed2, rng)
    
    if len(seed1_removed) < len(seed2_removed):
        current = seed1_removed
    else:
        current = seed2_removed
        
    best = list(current)
    best_len = len(best)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    
    while _time_left() > 5.0:
        # Local search on current
        current = greedy_remove(current, rng)
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
            T = 10.0
        
        # Generate neighbor
        if rng.random() < 0.3:
            # Smart add-remove: Add comparator that reduces inversions most
            neighbor = smart_add_remove(current, rng)
        else:
            # Random mutation
            neighbor = list(current)
            n = len(neighbor)
            op = rng.choice(['remove', 'swap', 'move', 'insert'])
            
            if op == 'remove' and n > 0:
                idx = rng.randrange(n)
                neighbor.pop(idx)
            elif op == 'swap' and n >= 2:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            elif op == 'move' and n >= 1:
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    elem = neighbor.pop(i1)
                    neighbor.insert(i2, elem)
            elif op == 'insert':
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i > j: i, j = j, i
                if i != j:
                    pos = rng.randrange(n + 1)
                    neighbor.insert(pos, (i, j))
            
            if verify_network(neighbor):
                neighbor = greedy_remove(neighbor, rng)
            else:
                neighbor = list(current)
        
        # Acceptance
        if len(neighbor) <= len(current):
            current = neighbor
        else:
            delta = len(neighbor) - len(current)
            if delta <= T:
                prob = 0.5 ** (delta / max(T, 1e-6))
                if rng.random() < prob:
                    current = neighbor
        
        T *= cooling
    
    best = greedy_remove(best, rng)
    return best

def smart_add_remove(comps, rng):
    """
    Add the comparator that most reduces inversions in the current state,
    then greedily remove.
    """
    if _time_left() < 2.0:
        return comps
    
    state = list(INITIAL_STATES)
    # Apply current comps to get state
    for i, j in comps:
        wi = state[i]
        wj = state[j]
        state[i] = wi & wj
        state[j] = wi | wj
        
    current_inv = calc_inversions(state)
    
    best_delta = 0
    best_comp = None
    
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in comps:
                continue
            new_state = list(state)
            wi = new_state[i]
            wj = new_state[j]
            new_state[i] = wi & wj
            new_state[j] = wi | wj
            new_inv = calc_inversions(new_state)
            delta = new_inv - current_inv
            if delta < best_delta:
                best_delta = delta
                best_comp = (i, j)
                
    if best_comp is None:
        return comps
        
    n = len(comps)
    pos = rng.randrange(n + 1)
    candidate = comps[:pos] + [best_comp] + comps[pos:]
    candidate = greedy_remove(candidate, rng)
    
    return candidate
