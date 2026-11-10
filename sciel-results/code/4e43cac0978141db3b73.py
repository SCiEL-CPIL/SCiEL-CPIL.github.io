def _simulate_annealing(sequence, dimension, rng, num_iterations=10000):
    """Simulated annealing with chain-reversal, single-step, and pivot moves."""
    n = len(sequence)
    if n <= 1:
        return _initial_spiral(sequence, dimension)
    
    # Start from a good initial conformation
    chain = _initial_spiral(sequence, dimension)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
    
    for it in range(num_iterations):
        T = T_start * (cooling_rate ** it)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5 and n > 3:
            # Pivot move
            i = rng.randint(1, n - 2)
            max_L = n - 1 - i
            L = rng.randint(1, max_L)
            
            pivot_pos = chain[i]
            seg_indices = list(range(i + 1, i + L + 1))
            
            valid_move = None
            for angle in (1, 2, 3):
                new_positions = {}
                valid = True
                for k in seg_indices:
                    old_pos = chain[k]
                    dx = old_pos[0] - pivot_pos[0]
                    dy = old_pos[1] - pivot_pos[1]
                    
                    if angle == 1:
                        new_dx, new_dy = -dy, dx
                    elif angle == 2:
                        new_dx, new_dy = -dx, -dy
                    else:
                        new_dx, new_dy = dy, -dx
                        
                    new_pos = (pivot_pos[0] + new_dx, pivot_pos[1] + new_dy)
                    
                    # Check if new_pos is occupied by a residue NOT in the segment
                    # The residue at new_pos must be in seg_indices OR new_pos must be empty
                    # But we are moving the segment, so the new_pos will be occupied by the segment.
                    # We must ensure that new_pos is NOT occupied by a residue OUTSIDE the segment.
                    # i.e., if new_pos is in occupied, then occupied[new_pos] must be in seg_indices.
                    
                    if new_pos in occupied:
                        if occupied[new_pos] not in seg_indices:
                            valid = False
                            break
                            
                    new_positions[k] = new_pos
                    
                if valid:
                    valid_move = (seg_indices, new_positions, angle)
                    break
            if valid_move:
                seg_indices, new_positions, angle = valid_move
                old_positions = {k: chain[k] for k in seg_indices}
                
                for k in seg_indices:
                    del occupied[chain[k]]
                    chain[k] = new_positions[k]
                    occupied[chain[k]] = k
                    
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    for k in seg_indices:
                        del occupied[chain[k]]
                        chain[k] = old_positions[k]
                        occupied[chain[k]] = k

        elif move_type < 0.8 and n > 4:
            # Chain reversal move
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            
            valid = True
            if i > 0:
                prev = chain[i - 1]
                new_first = chain[j]
                if abs(prev[0] - new_first[0]) + abs(prev[1] - new_first[1]) != 1:
                    valid = False
            
            if j < n - 1:
                next_pos = chain[j + 1]
                new_last = chain[i]
                if abs(next_pos[0] - new_last[0]) + abs(next_pos[1] - new_last[1]) != 1:
                    valid = False
            
            if valid:
                old_segment = chain[i:j + 1]
                chain[i:j + 1] = old_segment[::-1]
                
                # Update occupied
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
                    
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                else:
                    chain[i:j + 1] = old_segment
                    for k in range(i, j + 1):
                        occupied[chain[k]] = k

        else:
            # Single step move
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            candidates = []
            for q in _neighbours(old_pos):
                # q must be free or occupied by i
                if q not in occupied or occupied[q] == i:
                    # Check adjacency to i-1 and i+1
                    if abs(q[0] - chain[i-1][0]) + abs(q[1] - chain[i-1][1]) == 1 and \
                       abs(q[0] - chain[i+1][0]) + abs(q[1] - chain[i+1][1]) == 1:
                        candidates.append(q)
            
            if candidates:
                new_pos = rng.choice(candidates)
                if new_pos != old_pos:
                    chain[i] = new_pos
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    
                    new_energy = _compute_energy(sequence, chain)
                    
                    if new_energy < current_energy or rng.random() < math.exp((new_energy - current_energy) / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                    else:
                        chain[i] = old_pos
                        del occupied[new_pos]
                        occupied[old_pos] = i
    
    return best_chain
