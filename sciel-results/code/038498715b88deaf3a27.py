<<<<<<< SEARCH
def _local_search(comparators, deadline, rng):
    """
    Perform local search: try swaps, moves, and replacements.
    Returns improved comparators if found.
    """
    n_comps = len(comparators)
    if n_comps < 2:
        return comparators
    
    best = comparators[:]
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        move_type = rng.randint(0, 2)
        idx = rng.randint(0, len(best) - 1)
        
        if move_type == 0:
            # Swap with another position
            other = rng.randint(0, len(best) - 1)
            if other == idx:
                continue
            candidate = best[:]
            candidate[idx], candidate[other] = candidate[other], candidate[idx]
            
        elif move_type == 1:
            # Move to different position
            new_idx = rng.randint(0, len(best) - 1)
            if new_idx == idx:
                continue
            candidate = best[:]
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
        else:
            # Replace with random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate = best[:]
            candidate[idx] = (i, j)
        
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                best = candidate
    
    return best
=======
def _local_search(comparators, deadline, rng):
    """
    Perform aggressive local search: try deletions, swaps, and moves.
    This is the core of the Iterated Local Search strategy.
    """
    best = comparators[:]
    n_comps = len(best)
    
    # If we have time, we can try many moves
    # We prioritize deletions, but allow swaps/moves to escape local minima
    while time.time() < deadline:
        if n_comps < 2:
            break
        
        move_type = rng.random()
        idx = rng.randint(0, n_comps - 1)
        
        if move_type < 0.4:
            # Try deletion (40% probability)
            candidate = best[:]
            candidate.pop(idx)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    best = candidate
                    n_comps -= 1
                    continue
        elif move_type < 0.8:
            # Try swap (40% probability)
            other = rng.randint(0, n_comps - 1)
            if other == idx:
                continue
            candidate = best[:]
            candidate[idx], candidate[other] = candidate[other], candidate[idx]
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    best = candidate
                    # Swaps don't change length, so no update to n_comps needed
                    # But we accept it to change the structure
        else:
            # Try move (20% probability)
            new_idx = rng.randint(0, n_comps - 1)
            if new_idx == idx:
                continue
            candidate = best[:]
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    best = candidate
    
    return best

def _perturb(comparators, rng, strength=5):
    """
    Randomly perturb the network to escape local minima.
    """
    perturbed = comparators[:]
    n_comps = len(perturbed)
    if n_comps < 2:
        return perturbed
    
    for _ in range(strength):
        if time.time() > 88.0: # Hard deadline guard for perturb
            break
        move_type = rng.randint(0, 1)
        idx = rng.randint(0, n_comps - 1)
        
        if move_type == 0:
            other = rng.randint(0, n_comps - 1)
            if other != idx:
                perturbed[idx], perturbed[other] = perturbed[other], perturbed[idx]
        else:
            new_idx = rng.randint(0, n_comps - 1)
            if new_idx != idx:
                comp = perturbed.pop(idx)
                perturbed.insert(new_idx, comp)
                
    return perturbed
>>>>>>> REPLACE
