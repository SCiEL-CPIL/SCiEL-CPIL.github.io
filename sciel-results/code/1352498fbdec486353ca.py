    # Initial Phase: Multi-start
    best_seq = greedy_construct() # Standard greedy
    best_len = len(best_seq)
    
    for _ in range(20): # 20 random greedy starts
        if time.time() > deadline:
            break
        seq = greedy_construct_shuffled(all_values)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    current_seq = best_seq
