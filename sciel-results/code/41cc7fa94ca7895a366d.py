<<<<<<< SEARCH
def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multi-start greedy to build initial population
    population = []
    
    while time.time() < deadline - 60:
        # Greedy Construction
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            seed_tset = random.choice(list(uncovered))
            block = list(seed_tset)
            
            while len(block) < k and time.time() < deadline - 5:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = frozenset(list(R) + [x])
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    if best_gain == 0:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = random.choice(remaining)
                        else:
                            break
                    else:
                        break
                else:
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                covered = set()
                for c in itertools.combinations(block, t):
                    covered.add(frozenset(c))
                uncovered -= covered
                blocks.append(block)
        
        # Check if we have full coverage
        if uncovered:
            for ts in list(uncovered):
                if time.time() >= deadline - 2:
                    break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
        
        if uncovered:
            continue
            
        # Prune
        pruned_blocks = _prune_redundant_blocks(blocks, v, t)
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 1:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t)
        
        # Apply local search swap
        if time.time() < deadline - 3:
            pruned_blocks = _local_search_swap(pruned_blocks, v, k, t, all_tsets, deadline - 2)
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
        
        population.append(pruned_blocks)
        
        # Keep population size manageable
        if len(population) >= 10:
            population.pop(0)
        
        if time.time() > deadline - 65:
            break

    # Phase 2: Genetic Algorithm evolution
    if population and time.time() < deadline - 10:
        # Sort population by size
        population.sort(key=lambda b: len(b))
        
        ga_deadline = deadline - 5
        generation = 0
        
        while time.time() < ga_deadline:
            generation += 1
            new_population = []
            
            # Elitism: keep top half
            elite_size = max(1, len(population) // 2)
            for i in range(min(elite_size, len(population))):
                new_population.append(population[i])
            
            # Generate offspring via crossover and mutation
            while len(new_population) < len(population) and time.time() < ga_deadline:
                # Select parents (tournament selection)
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                
                if random.random() < 0.6:
                    # Crossover
                    child = _ga_crossover(population[p1_idx], population[p2_idx], v, k, t, all_tsets, ga_deadline)
                else:
                    # Mutation
                    child = _ga_mutation(population[p1_idx], v, k, t, all_tsets, ga_deadline)
                
                if child:
                    # Prune the child
                    child = _prune_redundant_blocks(child, v, t)
                    prev_len = len(child)
                    while len(child) < prev_len and time.time() < ga_deadline - 1:
                        prev_len = len(child)
                        child = _prune_redundant_blocks(child, v, t)
                    
                    # Apply local search
                    if time.time() < ga_deadline - 2:
                        child = _local_search_swap(child, v, k, t, all_tsets, ga_deadline - 1)
                    
                    # Check validity
                    uncovered = set(all_tsets)
                    for b in child:
                        for c in itertools.combinations(b, t):
                            uncovered.discard(frozenset(c))
                    
                    if not uncovered:
                        new_population.append(child)
                        if len(child) < best_count:
                            best_count = len(child)
                            best_blocks = child
            
            if not new_population:
                break
            
            population = new_population
            population.sort(key=lambda b: len(b))
            
            # Stop if we've hit the known best
            if best_count <= 76:
                break

    # Final polish: apply local search to best solution
    if best_blocks is not None and time.time() < deadline - 3:
        polished = _local_search_swap(best_blocks, v, k, t, all_tsets, deadline - 1)
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished

    # Final fallback if no valid solution found
    if best_blocks is None:
        uncovered = set(all_tsets)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            seed_tset = next(iter(uncovered))
            block = list(seed_tset)
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set: continue
                    gain = sum(1 for R in itertools.combinations(block, t-1) if frozenset(list(R)+[x]) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem: best_x = rem[0]
                    else: break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                covered = set(frozenset(c) for c in itertools.combinations(block, t))
                uncovered -= covered
                blocks.append(block)
        
        if not uncovered:
            best_blocks = blocks
        else:
            for ts in list(uncovered):
                if time.time() >= deadline - 0.5: break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v-1)
                    if x not in block: block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
            best_blocks = blocks

    return best_blocks
=======
def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute triple mapping for O(1) lookup
    triple_map = _build_triple_index(v, t)
    total_tsets = len(triple_map)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multi-start greedy to build initial population
    population = []
    
    while time.time() < deadline - 40:
        # Greedy Construction using numpy for coverage
        coverage = np.zeros(total_tsets, dtype=np.int32)
        blocks = []
        
        while time.time() < deadline - 5:
            # Find an uncovered triple
            uncovered_mask = coverage == 0
            if not np.any(uncovered_mask):
                break
            
            uncovered_idxs = np.where(uncovered_mask)[0]
            seed_idx = random.choice(uncovered_idxs)
            
            # Reverse map index to triple
            # For t=3, we need to find the triple
            # Build reverse mapping on the fly for this v,t
            if t == 3:
                # Find triple with this index
                # idx = number of triples (a',b',c') with a'<a or (a'=a and b'<b) or (a'=a and b'=b and c'<c)
                # This is complex; let's just iterate
                seed_triple = None
                for a in range(v - 2):
                    for b in range(a + 1, v - 1):
                        for c in range(b + 1, v):
                            if triple_map[(a, b, c)] == seed_idx:
                                seed_triple = (a, b, c)
                                break
                        if seed_triple:
                            break
                    if seed_triple:
                        break
                block = list(seed_triple)
            else:
                # Generic case
                for combo, idx in triple_map.items():
                    if idx == seed_idx:
                        block = list(combo)
                        break
            
            # Extend block greedily
            while len(block) < k and time.time() < deadline - 5:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: how many new triples does adding x cover?
                    gain = 0
                    if t == 3:
                        for a in block:
                            for b in block:
                                if a >= b:
                                    continue
                                ts = tuple(sorted([a, b, x]))
                                if ts in triple_map and coverage[triple_map[ts]] == 0:
                                    gain += 1
                    else:
                        for combo in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(list(combo) + [x]))
                            if ts in triple_map and coverage[triple_map[ts]] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                else:
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                # Update coverage
                idxs = _block_triples_indices(block, v, t, triple_map)
                for idx in idxs:
                    coverage[idx] += 1
                blocks.append(block)
        
        # Check if we have full coverage
        if np.any(coverage == 0):
            # Try to repair
            for _ in range(50):
                if time.time() >= deadline - 2:
                    break
                new_block = sorted(random.sample(range(v), k))
                if new_block not in blocks:
                    blocks.append(new_block)
                    idxs = _block_triples_indices(new_block, v, t, triple_map)
                    for idx in idxs:
                        coverage[idx] += 1
                if not np.any(coverage == 0):
                    break
        
        if np.any(coverage == 0):
            continue
        
        # Prune
        pruned_blocks = _prune_redundant_blocks(blocks, v, t, triple_map)
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 1:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t, triple_map)
        
        # Apply local search swap
        if time.time() < deadline - 3:
            pruned_blocks = _local_search_swap(pruned_blocks, v, k, t, triple_map, deadline - 2)
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
        
        population.append(pruned_blocks)
        
        # Keep population size manageable
        if len(population) >= 8:
            population.pop(0)

    # Phase 2: Genetic Algorithm evolution
    if population and time.time() < deadline - 10:
        # Sort population by size
        population.sort(key=lambda b: len(b))
        
        ga_deadline = deadline - 5
        generation = 0
        
        while time.time() < ga_deadline:
            generation += 1
            new_population = []
            
            # Elitism: keep top half
            elite_size = max(1, len(population) // 2)
            for i in range(min(elite_size, len(population))):
                new_population.append(population[i])
            
            # Generate offspring via crossover and mutation
            while len(new_population) < len(population) and time.time() < ga_deadline:
                # Select parents (tournament selection)
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                
                if random.random() < 0.6:
                    # Crossover
                    child = _ga_crossover(population[p1_idx], population[p2_idx], v, k, t, triple_map, ga_deadline)
                else:
                    # Mutation
                    child = _ga_mutation(population[p1_idx], v, k, t, triple_map, ga_deadline)
                
                if child:
                    # Prune the child
                    child = _prune_redundant_blocks(child, v, t, triple_map)
                    prev_len = len(child)
                    while len(child) < prev_len and time.time() < ga_deadline - 1:
                        prev_len = len(child)
                        child = _prune_redundant_blocks(child, v, t, triple_map)
                    
                    # Apply local search
                    if time.time() < ga_deadline - 2:
                        child = _local_search_swap(child, v, k, t, triple_map, ga_deadline - 1)
                    
                    # Check validity
                    coverage_check = np.zeros(total_tsets, dtype=np.int32)
                    for b in child:
                        idxs = _block_triples_indices(b, v, t, triple_map)
                        for idx in idxs:
                            coverage_check[idx] += 1
                    
                    if not np.any(coverage_check == 0):
                        new_population.append(child)
                        if len(child) < best_count:
                            best_count = len(child)
                            best_blocks = child
            
            if not new_population:
                break
            
            population = new_population
            population.sort(key=lambda b: len(b))
            
            # Stop if we've hit the known best
            if best_count <= 76:
                break

    # Final polish: apply local search to best solution
    if best_blocks is not None and time.time() < deadline - 3:
        polished = _local_search_swap(best_blocks, v, k, t, triple_map, deadline - 1)
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished

    # Final fallback if no valid solution found
    if best_blocks is None:
        coverage = np.zeros(total_tsets, dtype=np.int32)
        blocks = []
        while time.time() < deadline - 1:
            uncovered_mask = coverage == 0
            if not np.any(uncovered_mask):
                break
            
            uncovered_idxs = np.where(uncovered_mask)[0]
            seed_idx = random.choice(uncovered_idxs)
            
            # Find triple
            if t == 3:
                seed_triple = None
                for a in range(v - 2):
                    for b in range(a + 1, v - 1):
                        for c in range(b + 1, v):
                            if triple_map[(a, b, c)] == seed_idx:
                                seed_triple = (a, b, c)
                                break
                        if seed_triple:
                            break
                    if seed_triple:
                        break
                block = list(seed_triple)
            else:
                for combo, idx in triple_map.items():
                    if idx == seed_idx:
                        block = list(combo)
                        break
            
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    if t == 3:
                        for a in block:
                            for b in block:
                                if a >= b:
                                    continue
                                ts = tuple(sorted([a, b, x]))
                                if ts in triple_map and coverage[triple_map[ts]] == 0:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem:
                        best_x = rem[0]
                    else:
                        break
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                idxs = _block_triples_indices(block, v, t, triple_map)
                for idx in idxs:
                    coverage[idx] += 1
                blocks.append(block)
        
        if not np.any(coverage == 0):
            best_blocks = blocks
        else:
            # Last resort: add random blocks
            for _ in range(100):
                if time.time() >= deadline - 0.5:
                    break
                new_block = sorted(random.sample(range(v), k))
                if new_block not in blocks:
                    blocks.append(new_block)
                    idxs = _block_triples_indices(new_block, v, t, triple_map)
                    for idx in idxs:
                        coverage[idx] += 1
                if not np.any(coverage == 0):
                    break
            best_blocks = blocks

    return best_blocks
>>>>>>> REPLACE
