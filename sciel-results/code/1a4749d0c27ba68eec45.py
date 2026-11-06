<<<<<<< SEARCH
    # Local Search Polish
    # We take the best_set found by DFS and try to improve it via 1-swaps
    
    current_set = list(best_set)
    current_set_set = set(current_set)
    current_score = best_score_float
    
    # Precompute a list of all elements 1..N for iteration
    all_elements = list(range(1, N + 1))
    
    improved = True
    # Limit local search time to 5 seconds
    ls_deadline = start_time + 50.0 
    
    while improved and time.time() < ls_deadline:
        improved = False
        # Iterate over elements in current_set to remove
        for x in current_set:
            if time.time() > ls_deadline:
                break
            # Create a copy of the set without x for checking
            # We don't need to copy the list, just use a set difference for checks
            # But we need to check validity of adding y to (current_set - {x})
            
            # Optimization: If x is large, its weight is small.
            # Swapping x for y > x might not improve if weights are small.
            # But y < x might be better.
            
            for y in all_elements:
                if y == x or y in current_set_set:
                    continue
                    
                # Check if y can be added to current_set - {x}
                # This is valid if no pair (a, b) in ap_pairs[y] is in current_set - {x}
                valid = True
                for a, b in ap_pairs[y]:
                    # a and b must be in current_set, and neither is x
                    if (a in current_set_set) and (b in current_set_set):
                        # If x is one of them, it's removed, so the pair is broken
                        if x == a or x == b:
                            continue
                        else:
                            valid = False
                            break
                
                if valid:
                    new_score = current_score - weights_float[x] + weights_float[y]
                    if new_score > current_score + 1e-12:
                        # Perform swap
                        current_set.remove(x)
                        current_set.append(y)
                        current_set_set.remove(x)
                        current_set_set.add(y)
                        current_score = new_score
                        improved = True
                        break # Restart the outer loop
            if improved:
                break
                
    # Update best
    if current_score > best_score_float:
        best_score_float = current_score
        best_set = current_set

    return best_set
=======
    # Simulated Annealing Polish
    # We take the best_set found by DFS and try to improve it via SA
    
    current_set = list(best_set)
    current_set_set = set(current_set)
    current_score = best_score_float
    
    all_elements = list(range(1, N + 1))
    
    # SA Parameters
    T0 = 0.01
    T_end = 0.0001
    alpha = 0.999
    T = T0
    
    while T > T_end and time.time() < sa_deadline:
        # Pick a random element to remove
        if not current_set:
            break
        x = random.choice(current_set)
        
        # Pick a random element to add
        y = random.choice(all_elements)
        
        if x == y or y in current_set_set:
            continue
            
        # Check validity of swap: remove x, add y
        # Valid if y does not form AP with any pair in (current_set - {x})
        valid = True
        for a, b in ap_pairs[y]:
            if (a in current_set_set) and (b in current_set_set):
                if x != a and x != b:
                    valid = False
                    break
        
        if not valid:
            continue
            
        delta = weights_float[y] - weights_float[x]
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept move
            current_set.remove(x)
            current_set.append(y)
            current_set_set.remove(x)
            current_set_set.add(y)
            current_score += delta
            
            if current_score > best_score_float:
                best_score_float = current_score
                best_set = list(current_set)
                
        T *= alpha

    # Final check
    if current_score > best_score_float:
        best_score_float = current_score
        best_set = current_set

    return best_set
>>>>>>> REPLACE
