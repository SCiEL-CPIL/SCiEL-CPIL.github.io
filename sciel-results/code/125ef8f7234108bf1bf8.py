<<<<<<< SEARCH
    # Final local search: try to add any missing element
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for x in range(1, n+1):
            if not S_best[x] and is_valid_addition(x, S_best, n):
                S_best[x] = True
                best_count += 1
                improved = True
    
    best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    return best_set_list
=======
    # Final local search: try to add any missing element
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for x in range(1, n+1):
            if not S_best[x] and is_valid_addition_bitset(x, bitset_best, n):
                S_best[x] = True
                bitset_best |= (1 << x)
                best_count += 1
                improved = True
    
    # Swap-based local search: try swapping elements to increase density
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        elements_in_set = [i for i in range(1, n+1) if S_best[i]]
        elements_not_in_set = [i for i in range(1, n+1) if not S_best[i]]
        
        random.shuffle(elements_in_set)
        random.shuffle(elements_not_in_set)
        
        for x_add in elements_not_in_set:
            if time.time() - start_time > max_time:
                break
            for x_remove in elements_in_set:
                if time.time() - start_time > max_time:
                    break
                if swap_valid(x_add, x_remove, bitset_best, n):
                    # Perform the swap
                    S_best[x_remove] = False
                    S_best[x_add] = True
                    bitset_best &= ~(1 << x_remove)
                    bitset_best |= (1 << x_add)
                    improved = True
                    # Update lists
                    elements_in_set.remove(x_remove)
                    elements_in_set.append(x_add)
                    elements_not_in_set.remove(x_add)
                    elements_not_in_set.append(x_remove)
                    break  # Move to next x_add
    
    # Another round of pure additions
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for x in range(1, n+1):
            if not S_best[x] and is_valid_addition_bitset(x, bitset_best, n):
                S_best[x] = True
                bitset_best |= (1 << x)
                best_count += 1
                improved = True
    
    best_set_list = [i for i in range(1, n+1) if S_best[i]]
    
    return best_set_list
>>>>>>> REPLACE
