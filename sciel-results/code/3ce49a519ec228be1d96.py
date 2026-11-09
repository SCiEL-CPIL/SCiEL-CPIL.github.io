<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85
    universe = list(range(1, 23))
    best = []
    best_len = 0
    heuristics = ['largest', 'smallest', 'random', 'random', 'largest']
    for h in heuristics:
        if time.time() > deadline - 10:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 10)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 5)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 3)
        if len(result) > best_len:
            best_len = len(result)
            best = result
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 1)
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85
    universe = list(range(1, 23))
    best = []
    best_len = 0
    rng = random.Random(42)
    
    # Phase 1: Deterministic greedy with different heuristics
    heuristics = ['largest', 'smallest', 'random', 'random', 'largest']
    for h in heuristics:
        if time.time() > deadline - 30:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 30)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 25)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 20)
        if len(result) > best_len:
            best_len = len(result)
            best = result
    
    # Phase 2: Randomized local search with multiple restarts
    if time.time() < deadline - 5:
        attempts = 0
        while time.time() < deadline - 5:
            attempts += 1
            result = _randomized_local_search(universe, deadline - 5, rng)
            if len(result) > best_len:
                best_len = len(result)
                best = result
                # If we found a very large set, do a final polish
                if best_len >= 10:
                    in_set = set(best)
                    not_in_set = [x for x in universe if x not in in_set]
                    best = _try_add_elements(best, not_in_set, deadline - 2)
                    not_in_set = [x for x in universe if x not in set(best)]
                    best = _try_swaps(best, not_in_set, deadline - 2)
    
    # Final polish
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 1)
    return best
>>>>>>> REPLACE
