<<<<<<< SEARCH
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = full_local_search(code, deadline - 15)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search with varied perturbations
    seed = 1
    while time.time() < deadline - 8:
        # Different perturbation strategies
        strategy = seed % 4
        
        if strategy == 0:
            # Small targeted removal + repair
            k = random.randint(1, 2)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 1:
            # Medium targeted removal + repair
            k = random.randint(3, 4)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 2:
            # Larger targeted removal + repair
            k = random.randint(5, 7)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        else:
            # Swap-based perturbation: remove 2 low-unique, add 2 random, then repair
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 2:
                # Find low-unique codewords to remove
                counts_tmp = np.zeros(total_words, dtype=np.int32)
                for c in perturbed_code:
                    for w in neighbors[c]:
                        counts_tmp[w] += 1
                unique_scores = []
                for c in perturbed_code:
                    unique = sum(1 for w in neighbors[c] if counts_tmp[w] == 1)
                    unique_scores.append(unique)
                order = sorted(range(len(perturbed_code)), key=lambda i: unique_scores[i])
                remove_idx = order[:2]
                for idx in sorted(remove_idx, reverse=True):
                    perturbed_code.pop(idx)
                # Add 2 random codewords
                for _ in range(2):
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 8)
        
        # Local search on perturbed code
        perturbed_code = full_local_search(perturbed_code, deadline - 8)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Fresh restarts with different seeds
        if seed % 6 == 0 and time.time() < deadline - 25:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = full_local_search(fresh_code, deadline - 25)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
=======
    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = full_local_search(code, deadline - 15)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search with varied perturbations
    seed = 1
    while time.time() < deadline - 8:
        # Different perturbation strategies
        strategy = seed % 4
        
        if strategy == 0:
            # Small targeted removal + repair
            k = random.randint(1, 2)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 1:
            # Medium targeted removal + repair
            k = random.randint(3, 4)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 2:
            # Larger targeted removal + repair
            k = random.randint(5, 7)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        else:
            # Swap-based perturbation: remove 2 low-unique, add 2 random, then repair
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 2:
                counts_tmp = np.zeros(total_words, dtype=np.int32)
                for c in perturbed_code:
                    for w in neighbors[c]:
                        counts_tmp[w] += 1
                unique_scores = []
                for c in perturbed_code:
                    unique = sum(1 for w in neighbors[c] if counts_tmp[w] == 1)
                    unique_scores.append(unique)
                order = sorted(range(len(perturbed_code)), key=lambda i: unique_scores[i])
                remove_idx = order[:2]
                for idx in sorted(remove_idx, reverse=True):
                    perturbed_code.pop(idx)
                for _ in range(2):
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 8)
        
        # Local search on perturbed code
        perturbed_code = full_local_search(perturbed_code, deadline - 8)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Fresh restarts with different seeds
        if seed % 6 == 0 and time.time() < deadline - 25:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = full_local_search(fresh_code, deadline - 25)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
>>>>>>> REPLACE
