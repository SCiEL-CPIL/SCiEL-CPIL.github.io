<<<<<<< SEARCH
    # --- Phase 2: Local-search repair ---
    # Try to escape local minima by removing a comparator and then repairing.
    current = best_network[:]
    for iteration in range(10000): # Increased iterations due to speed
        if (time.time() - start_time) > DEADLINE - 5:
            break
        if len(current) <= 70:
            break
            
        # Random removal
        idx = random.randint(0, len(current) - 1)
        candidate = current[:]
        candidate.pop(idx)
        
        # If it's still valid, accept
        if verify_network(candidate):
            current = candidate
            if len(current) < best_size:
                best_network = current[:]
                best_size = len(current)
            continue
            
        # If broken, try to repair by inserting a comparator
        # We try a few random insertions
        repaired = False
        for _ in range(20): # Limit repair attempts to avoid slowing down
            if (time.time() - start_time) > DEADLINE - 5:
                break
            pos = random.randint(0, len(candidate))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Avoid duplicating existing comparator at same position? 
            # Simple approach: just insert.
            temp = candidate[:]
            temp.insert(pos, (i, j))
            
            if verify_network(temp):
                current = temp
                repaired = True
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                break
                
        if not repaired:
            # If we can't repair, keep current
            pass
=======
    # --- Phase 2: Structured local search ---
    # Use swaps and moves to explore the neighborhood more effectively
    current = best_network[:]
    n_iters = 0
    while (time.time() - start_time) < DEADLINE - 10:
        n_iters += 1
        if len(current) <= 70:
            break
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = current[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if verify_network(candidate):
                current = candidate
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                    
        elif move_type < 0.7:
            # Move a comparator to a different position
            if len(current) < 2:
                continue
            i = random.randint(0, len(current) - 1)
            j = random.randint(0, len(current) - 1)
            if i == j:
                continue
            candidate = current[:]
            comp = candidate.pop(i)
            candidate.insert(j, comp)
            if verify_network(candidate):
                current = candidate
                if len(current) < best_size:
                    best_network = current[:]
                    best_size = len(current)
                    
        else:
            # Try removing and re-inserting (sliding)
            if len(current) < 2:
                continue
            i = random.randint(0, len(current) - 1)
            candidate = current[:]
            comp = candidate.pop(i)
            # Try inserting at nearby positions
            for offset in [1, -1, 2, -2, 3, -3]:
                j = i + offset
                if 0 <= j <= len(candidate):
                    temp = candidate[:]
                    temp.insert(j, comp)
                    if verify_network(temp):
                        current = temp
                        if len(current) < best_size:
                            best_network = current[:]
                            best_size = len(current)
                        break
>>>>>>> REPLACE

