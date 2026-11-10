def _local_search_improve(S, primes, time_budget, randomize=True, rng=None):
    """
    Attempts to improve S by trying to add back elements that were removed.
    This is a local search where we try to add elements one by one and repair if needed.
    """
    if rng is None:
        rng = random
        
    S = S.copy()
    start_time = time.time()
    
    # Find all positions not in S
    candidate_positions = np.where(~S)[0]
    if len(candidate_positions) == 0:
        return S
    
    # Shuffle candidates
    rng.shuffle(candidate_positions)
    
    current_size = np.sum(S)
    
    for pos in candidate_positions:
        if time.time() - start_time > time_budget:
            break
            
        # Try adding pos
        S_test = S.copy()
        S_test[pos] = True
        new_size = current_size + 1
        
        # Check if still admissible
        is_adm = True
        indices = np.where(S_test)[0] # Compute once
        if len(indices) == 0:
            is_adm = True
        else:
            for p in primes:
                if p > new_size:
                    break
                residues = indices % p
                counts = np.bincount(residues, minlength=p)
                if np.all(counts > 0):
                    is_adm = False
                    break
        
        if is_adm:
            S = S_test
            current_size = new_size
        else:
            # Try to repair by removing some elements
            S_repaired = _greedy_repair(S_test, primes, randomize=randomize, rng=rng)
            if np.sum(S_repaired) > current_size:
                S = S_repaired
                current_size = np.sum(S)
                
    return S
