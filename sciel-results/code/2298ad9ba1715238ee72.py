def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    candidates = [(i, vectors[i]) for i in range(2, 41)]
    
    best_size = 0
    best_set = []
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        current_sums = {0}
        
        for idx, (val, vec) in enumerate(candidates):
            # Pruning
            if len(current_set) + (len(candidates) - idx) <= best_size:
                break
            
            valid = True
            new_sums = set()
            for s in current_sums:
                new_s = s + vec
                if new_s in current_sums:
                    valid = False
                    break
                new_sums.add(new_s)
            
            if valid:
                current_set.append(val)
                current_sums = current_sums | new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    
        if best_size >= 18:
            break

    return best_set
