    max_attempts = 3000
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        cand = random.randint(0, (1 << 17) - 1)
        if cand in code_set:
            attempts += 1
            continue
            
        new_cand = try_insert(cand, code_set, max_iters=50)
        if new_cand is not None:
            code_set.add(new_cand)
        
        attempts += 1
