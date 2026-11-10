<<<<<<< SEARCH
def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot moves only (Segment/Reversal were invalid).
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_end = deadline
    initial_temp = 5.0
    
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < t_end:
        # Only use Pivot moves as they preserve connectivity
        i = rng.randint(1, n - 1)
        result = _pivot_move(sequence, current_chain, i, rng)
        
        if result is None:
            # Failed move (collision), continue
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
            
        # Cooling
        temperature *= 0.999
        
        if temperature < 0.01:
            if no_improve_count > max_no_improve:
                temperature = initial_temp * 0.5
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.1
                no_improve_count = 0

    return best_chain, best_energy
=======
def _local_search(sequence, chain, rng, max_iters=500):
    """
    Local search: try all pivot positions and accept only improving moves.
    Returns (improved_chain, improved_energy).
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    improved = False
    for _ in range(max_iters):
        best_move = None
        best_delta = 0
        
        for i in range(1, n):
            result = _pivot_move(sequence, current_chain, i, rng)
            if result is not None:
                new_chain, delta = result
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, new_chain, delta)
        
        if best_move is None or best_delta >= 0:
            break
        
        _, current_chain, current_energy_delta = best_move
        current_energy += current_energy_delta
        improved = True
    
    return current_chain, current_energy

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot and Segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_end = deadline
    initial_temp = 3.0
    
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 3000
    step_count = 0
    
    while time.time() < t_end:
        # Choose move type: 70% pivot, 30% segment
        if rng.random() < 0.7:
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng)
        else:
            # Segment move: pick random segment length
            seg_len = rng.randint(2, min(10, n - 1))
            i = rng.randint(0, n - 1 - seg_len)
            j = i + seg_len
            result = _segment_move(sequence, current_chain, i, j, rng)
        
        if result is None:
            # Failed move (collision), continue
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta <= 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy += delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
        
        step_count += 1
        
        # Cooling: geometric with restart
        temperature *= 0.9995
        
        if temperature < 0.05:
            if no_improve_count > max_no_improve:
                # Restart with higher temperature
                temperature = initial_temp * 0.8
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.3
                no_improve_count = 0

    return best_chain, best_energy
>>>>>>> REPLACE

