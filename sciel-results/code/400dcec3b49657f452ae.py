<<<<<<< SEARCH
def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation
        if rng.random() < 0.6:
            init_chain, init_energy = _greedy_grow(sequence, rng)
            if init_chain is None:
                init_chain = _random_walk(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
        else:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        
        # Run local search with a portion of the time
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        # Polish pass with remaining time
        if time.time() < deadline - 0.1:
            polish_deadline = min(restart_deadline - 0.02, deadline - 0.1)
            if polish_deadline > time.time() + 0.05:
                final_chain, final_energy = _polish_pass(sequence, ls_chain, polish_deadline, rng)
                if final_energy < ls_energy:
                    ls_chain, ls_energy = final_chain, final_energy
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy
=======
def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline - 0.05:
            break
        
        # Generate initial conformation
        if rng.random() < 0.55:
            init_chain, init_energy = _greedy_grow(sequence, rng)
            if init_chain is None:
                init_chain = _random_walk(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
        else:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        
        # Run local search with a portion of the time
        ls_deadline = restart_deadline - 0.08
        if time.time() < ls_deadline:
            ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        else:
            ls_chain, ls_energy = list(init_chain), init_energy
        
        # Polish pass with remaining time
        polish_deadline = min(restart_deadline - 0.02, deadline - 0.1)
        if time.time() < polish_deadline - 0.05:
            final_chain, final_energy = _polish_pass(sequence, ls_chain, polish_deadline, rng)
            if final_energy < ls_energy:
                ls_chain, ls_energy = final_chain, final_energy
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = list(ls_chain)
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy
>>>>>>> REPLACE

