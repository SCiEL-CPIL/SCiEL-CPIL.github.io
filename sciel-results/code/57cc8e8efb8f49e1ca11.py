<<<<<<< SEARCH
def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    sa_time = remaining * 0.85
    polish_time = remaining * 0.05
    
    if sa_time < 1.0:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    if n <= 30:
        num_restarts = 4
    elif n <= 50:
        num_restarts = 3
    else:
        num_restarts = 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 10.0:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    remaining_for_polish = _TOTAL_BUDGET - 3.0 - (time.time() - _GLOBAL_START)
    if remaining_for_polish > 1.0:
        polish_rounds = 300 if n <= 40 else 200
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 3.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]
=======
def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    remaining = _TOTAL_BUDGET - elapsed
    sa_time = remaining * 0.88
    polish_time = remaining * 0.07
    
    if sa_time < 1.0:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # More restarts for shorter sequences, fewer for longer
    if n <= 25:
        num_restarts = 5
    elif n <= 40:
        num_restarts = 4
    elif n <= 60:
        num_restarts = 3
    else:
        num_restarts = 2
    time_per_restart = sa_time / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - 8.0:
            break
            
        init_rng = random.Random(12345 + n + k * 137)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        sa_rng = random.Random(12345 + n + k * 137 + 7)
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_per_restart, sa_rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    remaining_for_polish = _TOTAL_BUDGET - 2.0 - (time.time() - _GLOBAL_START)
    if remaining_for_polish > 0.5:
        polish_rounds = 500 if n <= 30 else (300 if n <= 50 else 200)
        polished_chain, polished_energy = _greedy_polish(sequence, best_chain, max_rounds=polish_rounds, deadline=_TOTAL_BUDGET - 2.0)
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
            
    return [list(p) for p in best_chain]
>>>>>>> REPLACE
