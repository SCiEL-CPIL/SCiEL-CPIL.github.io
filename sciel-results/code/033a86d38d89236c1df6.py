    # Strategy 3: Random restarts with local search
    random.seed(42)
    count = 0
    while time.time() - start_time < deadline - 8:
        # Random greedy
        order = list(range(1, N+1))
        random.shuffle(order)
        S_r = greedy_add_order(order)
        S_improved = local_search_fast(S_r, max_iter=2000)
        
        if len(S_improved) > best_size:
            best = S_improved
            best_size = len(S_improved)
        
        count += 1
        if count > 500:
            break
