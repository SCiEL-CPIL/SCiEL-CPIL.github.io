<<<<<<< SEARCH
def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Generate diverse initial seeds
    seeds = []
    
    # Cantor
    cantor = _get_initial_cantor()
    seeds.append(cantor)
    
    # Forward greedy
    fwd = _greedy_forward()
    seeds.append(fwd)
    
    # Reverse greedy
    rev = _greedy_reverse()
    seeds.append(rev)
    
    # Random greedy (multiple)
    rng = random.Random(42)
    for _ in range(5):
        g = _greedy_random(rng)
        seeds.append(g)
    
    # Weighted greedy (multiple)
    for _ in range(3):
        g = _greedy_weighted(rng)
        seeds.append(g)
    
    best_bits = 0
    best_size = 0
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Greedy fill
        filled = _greedy_fill(seed, deadline - 10)
        size = bin(filled).count('1')
        
        if size > best_size:
            best_bits = filled
            best_size = size
    
    # Phase 2: Hill climbing on best seed
    if time.time() < deadline - 5:
        hc_bits, hc_size = _hill_climbing(best_bits, deadline - 5, random.Random(123))
        if hc_size > best_size:
            best_bits = hc_bits
            best_size = hc_size
    
    # Phase 3: Pair swap local search
    if time.time() < deadline - 3:
        ps_bits, ps_size = _pair_swap_local_search(best_bits, deadline - 3, random.Random(456))
        if ps_size > best_size:
            best_bits = ps_bits
            best_size = ps_size
    
    # Phase 4: Simulated annealing for additional exploration
    if time.time() < deadline - 1:
        sa_bits, sa_size = _simulated_annealing(best_bits, deadline - 1, random.Random(789))
        if sa_size > best_size:
            best_bits = sa_bits
            best_size = sa_size
    
    # Final greedy fill
    if time.time() < deadline - 1:
        best_bits = _greedy_fill(best_bits, deadline - 1)
    
    # Convert to set
    best_set = {i for i in range(1, N + 1) if (best_bits >> i) & 1}
    
    # Verification
    if _count_violations(best_bits) != 0:
        # Fallback to Cantor
        best_set = {i for i in range(1, N + 1) if (1 << i) & _get_initial_cantor()}
    
    return sorted(best_set)
=======
def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Generate diverse initial seeds
    seeds = []
    rng = random.Random(42)
    
    # Cantor
    seeds.append(_get_initial_cantor())
    
    # Forward/reverse greedy
    seeds.append(_greedy_forward())
    seeds.append(_greedy_reverse())
    
    # Multiple random greedy with different seeds
    for s in [1, 7, 13, 23, 37, 53, 71, 89]:
        seeds.append(_greedy_random(random.Random(s)))
    
    # Weighted greedy
    for s in [5, 11, 19]:
        seeds.append(_greedy_weighted(random.Random(s)))
    
    best_bits = 0
    best_size = 0
    
    # Phase 1: Greedy fill on all seeds, keep best
    for seed in seeds:
        if time.time() > deadline - 25:
            break
        filled = _greedy_fill(seed, deadline - 25)
        size = bin(filled).count('1')
        if size > best_size:
            best_bits = filled
            best_size = size
    
    # Phase 2: Fast local search on best
    if time.time() < deadline - 15:
        ls_bits, ls_size = _fast_local_search(best_bits, deadline - 15, random.Random(123))
        if ls_size > best_size:
            best_bits = ls_bits
            best_size = ls_size
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 8:
        sa_bits, sa_size = _simulated_annealing(best_bits, deadline - 8, random.Random(789))
        if sa_size > best_size:
            best_bits = sa_bits
            best_size = sa_size
    
    # Phase 4: Another fast local search round
    if time.time() < deadline - 3:
        ls2_bits, ls2_size = _fast_local_search(best_bits, deadline - 3, random.Random(456))
        if ls2_size > best_size:
            best_bits = ls2_bits
            best_size = ls2_size
    
    # Final greedy fill
    if time.time() < deadline - 1:
        best_bits = _greedy_fill(best_bits, deadline - 1)
    
    # Convert to set
    best_set = {i for i in range(1, N + 1) if (best_bits >> i) & 1}
    
    # Verification
    if _count_violations(best_bits) != 0:
        best_set = {i for i in range(1, N + 1) if (1 << i) & _get_initial_cantor()}
    
    return sorted(best_set)
>>>>>>> REPLACE
