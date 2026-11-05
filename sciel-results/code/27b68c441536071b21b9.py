def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    all_numbers = list(range(1, 4097))
    
    while time.time() < deadline:
        sample_size = random.choice([100, 200, 500, 1000])
        candidates = random.sample(all_numbers, sample_size)
        random.shuffle(candidates)
        
        current_set = []
        current_set_items = set()
        
        for x in candidates:
            if x in current_set_items:
                continue
            
            if is_valid_b3_incremental(current_set, x):
                current_set.append(x)
                current_set_items.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
