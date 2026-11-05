<<<<<<< SEARCH
def _greedy_polish(sequence, chain, max_rounds=100, deadline=None):
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy
=======
def _greedy_polish(sequence, chain, max_rounds=100, deadline=None):
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy

def _mwis_guided_polish(sequence, chain, time_limit, rng):
    """Contact-guided local search: identify missing H-H contacts (MWIS nodes) and
    try targeted moves to realize them while maintaining self-avoidance."""
    n = len(chain)
    if n < 6:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    max_iters = 50000
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Identify potential contacts to realize (sorted by proximity)
        potential = _potential_hh_pairs(sequence, current_chain)
        if not potential:
            break
        
        # Try to realize the closest potential contacts
        found_improvement = False
        for i, j, dist in potential[:20]:
            if time.time() - start_time > time_limit:
                break
            
            # Try pivot moves near residues i and j to bring them closer
            for k in [max(0, i-1), i, min(n-2, i+1), max(0, j-1), j, min(n-2, j+1)]:
                if k >= n - 1:
                    continue
                for rot_choice in range(3):
                    new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                    if new_chain is not None and new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        found_improvement = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try crankshaft moves in the region between i and j
            if n > 8:
                min_k = max(0, min(i, j) - 2)
                max_k = min(n - 1, max(i, j) + 2)
                for k1 in range(min_k, min(n - 2, max_k)):
                    for k2 in range(k1 + 2, max_k + 1):
                        for rot_choice in range(3):
                            new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                            if new_chain is not None and new_energy < current_energy:
                                current_chain = new_chain
                                current_energy = new_energy
                                found_improvement = True
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_chain = list(current_chain)
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
            
            if not found_improvement:
                # No improvement possible with current moves; break
                break
    
    return best_chain, best_energy
>>>>>>> REPLACE

