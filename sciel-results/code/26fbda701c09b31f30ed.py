    # Initial Local Search
    # Use a small fraction or just run it, it's usually fast
    best_chosen, best_mask, best_score = local_search(chosen, chosen_mask, get_score(chosen), deadline)
    
    # Iterated Local Search
    random.seed(42)
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Limit time for this restart to e.g., 10% of remaining or 5s
        ls_time_limit = time.time() + min(5.0, remaining * 0.5)
        
        # Perturb: remove 2 random elements
        perturbed = list(best_chosen)
        if len(perturbed) < 3:
            break
            
        idx1 = random.randint(0, len(perturbed) - 1)
        idx2 = random.randint(0, len(perturbed) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(perturbed) - 1)
            
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            
        perturbed.pop(idx1)
        perturbed.pop(idx2)
        
        p_mask = 0
        p_score = 0.0
        for a in perturbed:
            p_mask |= (1 << a)
            p_score += w[a]
            
        new_chosen, new_mask, new_score = local_search(perturbed, p_mask, p_score, ls_time_limit)
        
        if new_score > best_score:
            best_chosen = new_chosen
            best_mask = new_mask
            best_score = new_score

    return best_chosen
