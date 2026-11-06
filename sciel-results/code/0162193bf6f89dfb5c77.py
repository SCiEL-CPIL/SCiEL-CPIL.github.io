<<<<<<< SEARCH
    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 10:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from best initials
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 5:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 5 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(2, min(remaining_time / remaining_starts, 25))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 10 and best_A is not None:
        num_extra = 3
        for j in range(num_extra):
            if time.time() >= deadline - 5:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)
            
            time_alloc = max(2, (deadline - 5 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
=======
    # Precompute partner table for faster count updates
    partner_table = precompute_partners()
    
    # Multi-start: try each initial solution with local search
    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(initials)
    
    # Phase 1: Quick evaluation of all initials
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 8:
            break
        s, c = evaluate(init)
        cost = s if c else s + 10000
        scored_initials.append((cost, init))
    
    # Sort by cost (best first)
    scored_initials.sort(key=lambda x: x[0])
    
    # Phase 2: Local search from best initials
    for i, (cost, init) in enumerate(scored_initials):
        if time.time() >= deadline - 8:
            break
        remaining_starts = len(scored_initials) - i
        remaining_time = deadline - 8 - time.time()
        if remaining_time <= 0:
            break
        time_alloc = max(3, min(remaining_time / remaining_starts, 30))
        
        seed = 1000 + i
        ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=seed, partner_table=partner_table)
        
        if ls_score < best_score:
            best_score = ls_score
            best_A = ls_result
    
    # Phase 3: Additional random restarts if time remains
    if time.time() < deadline - 15 and best_A is not None:
        num_extra = 4
        for j in range(num_extra):
            if time.time() >= deadline - 8:
                break
            random.seed(2000 + j)
            size = random.randint(40, 90)
            init = set(random.sample(range(0, 129), size))
            init = fix_coverage(init)
            
            time_alloc = max(3, (deadline - 8 - time.time()) / (num_extra - j))
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed=3000 + j, partner_table=partner_table)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
>>>>>>> REPLACE
