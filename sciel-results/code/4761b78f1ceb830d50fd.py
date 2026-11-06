def _local_search(selected, conflicts, time_limit):
    start_time = time.time()
    current_set = set(selected)
    current_size = len(current_set)
    best_set = set(selected)
    best_size = len(best_set)
    
    temp = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # 1. Try to add any valid element
        improved = False
        # Optimization: Only check elements not in set
        # We can cache which elements are blocked?
        # For now, simple check.
        for x in range(1, 129):
            if x not in current_set and _can_add(x, current_set, conflicts):
                current_set.add(x)
                current_size += 1
                improved = True
        
        if improved:
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
            continue
        
        # 2. Kick: Remove random elements
        # Choose kick size
        kick_size = random.randint(1, 3)
        if current_size < kick_size:
            kick_size = current_size
            
        if kick_size == 0:
            # Stuck with empty or tiny set? Just add something
            for x in range(1, 129):
                if _can_add(x, current_set, conflicts):
                    current_set.add(x)
                    current_size += 1
                    break
            continue

        # Save state to revert if needed
        prev_set = set(current_set)
        prev_size = current_size
        
        items_to_remove = random.sample(list(current_set), kick_size)
        for item in items_to_remove:
            current_set.remove(item)
        current_size -= kick_size
        
        # Greedy Fill
        current_set = _greedy_fill(current_set, conflicts)
        current_size = len(current_set)
        
        # Acceptance
        if current_size > best_size:
            best_set = set(current_set)
            best_size = current_size
        elif current_size == best_size:
            # Accept to explore
            pass
        else:
            # Worse solution
            if random.random() < temp:
                # Accept
                pass
            else:
                # Reject, revert
                current_set = prev_set
                current_size = prev_size
        
        temp *= cooling
        if temp < 0.05:
            temp = 0.05

    return best_set
