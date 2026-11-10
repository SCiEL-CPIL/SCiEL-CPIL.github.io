def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Growth phase
    if n <= 48:
        num_growths = 200
    elif n <= 80:
        num_growths = 100
    else:
        num_growths = 50
        
    growth_limit = time.time() + 2.0 # Max 2 seconds for growth
    for attempt in range(num_growths):
        if time.time() > growth_limit:
            break
        if _remaining_time() < 5.0:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
            
    # SA phase: Run multiple rounds if time permits
    if best is not None:
        # Each SA run should take a reasonable chunk of time
        # Let's run SA until 1 second is left, or for a fixed number of iterations
        # We'll just pass the remaining time to SA.
        # But if we call SA multiple times, we need to split the time.
        
        # Strategy: Run SA for (remaining_time - 1) / 2, then again?
        # Or just one long SA run?
        # One long SA run is better for convergence.
        
        sa_time_limit = time.time() + _remaining_time() - 1.0
        improved_chain = _simulated_annealing(sequence, best, sa_time_limit, rng)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best, best_energy = improved_chain, improved_energy
            
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best]
