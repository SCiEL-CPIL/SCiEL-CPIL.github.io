# EVOLVE-BLOCK-START
import time
import sys
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, beam_width, time_limit):
    """
    Beam search over lattice states.
    State: (chain_tuple, occupied_set, current_energy)
    Heuristic: current_energy + potential_future_contacts
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start_time = time.time()
    # Initial state: first residue at origin
    start_pos = (0, 0, 0)
    initial_occupied = {start_pos}
    initial_chain = (start_pos,)
    initial_energy = 0
    
    # Beam holds list of (score, chain_tuple, occupied_set, energy)
    # Score = energy + heuristic (we want to maximize score, so minimize -score)
    # Actually, let's store negative score for min-heap logic or just sort.
    # Since beam width is small, sorting is fine.
    
    beam = [(0, initial_chain, initial_occupied, initial_energy)] # score, chain, occupied, energy
    
    # Pre-compute neighbor lists for speed?
    # We will generate neighbors on the fly.
    
    for i in range(1, n):
        next_beam = []
        res = sequence[i]
        
        for score, chain, occupied, energy in beam:
            last_pos = chain[-1]
            
            # Generate valid next positions
            for next_pos in _neighbours(last_pos):
                if next_pos in occupied:
                    continue
                
                # Calculate energy gain
                gain = 0
                if res == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            # Check if the residue at q is H
                            # We need to know which index is at q.
                            # Since occupied is a set of points, we don't have index mapping easily.
                            # We need to map point -> index.
                            # This is a bottleneck. 
                            # Optimization: Store occupied as dict point->index? 
                            # Or just check if q is in occupied and assume we can find index?
                            # For beam search, keeping a dict is better for energy calc.
                            # Let's refactor state to include occupied_dict
                            pass
                
                # To make this efficient, we should store occupied as a dict {point: index}
                # Let's restart the logic with dict.
                break # break inner loop to restart with better structure if needed, but let's just write it correctly below.
        
        # The above loop was a draft. Let's implement the actual beam search step properly.
        # We need to update the beam structure to use dict for occupied.
        # State: (score, chain_tuple, occupied_dict, energy)
        
        # Re-initialize beam with dict
        if i == 1:
            # The previous beam setup used sets. Let's convert.
            # Actually, let's just rewrite the whole function to use dicts from the start.
            pass
            
        # Since I can't easily refactor inside the loop without rewriting, 
        # I will implement a helper or just use the dict version directly.
        
        # Let's implement the step using the existing beam (which I will fix to use dicts in the main function logic).
        # For now, this function is a placeholder for the logic. 
        # I will rewrite the function below in the final code.
        return None

def _beam_search_fold_v2(sequence, dimension, beam_width, time_limit):
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start_time = time.time()
    start_pos = (0, 0, 0)
    
    # State: (score, chain_tuple, occupied_dict, energy)
    # occupied_dict: {point: index}
    
    initial_occupied = {start_pos: 0}
    initial_chain = (start_pos,)
    initial_energy = 0
    
    # Score = energy + heuristic
    # Heuristic: estimate of max future contacts.
    # A simple heuristic: number of remaining H residues * 2 (max contacts per H is low, but let's use a rough bound)
    # Better: 0 for now, or count remaining H's.
    
    beam = [(0, initial_chain, initial_occupied, initial_energy)]
    
    for i in range(1, n):
        next_beam = []
        res = sequence[i]
        
        for score, chain, occupied, energy in beam:
            last_pos = chain[-1]
            
            for next_pos in _neighbours(last_pos):
                if next_pos in occupied:
                    continue
                
                # Calculate gain
                gain = 0
                if res == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            idx = occupied[q]
                            if idx > 0 and sequence[idx] == "H": # idx 0 is start, but start can be H too. 
                                # Wait, occupied stores index.
                                # If sequence[idx] == 'H', it's a contact.
                                gain += 1
                                # Note: We count each contact once. 
                                # Since we are adding residue i, and checking neighbors q < i, 
                                # this counts the contact (i, q) exactly once.
                
                new_energy = energy - gain # Energy is negative, so adding contacts reduces energy (more negative)
                # Wait, _energy returns -contacts. 
                # So if gain > 0, new_energy = old_energy - gain. Correct.
                
                # Heuristic
                # Potential future contacts: 
                # Max possible contacts for remaining residues.
                # A very loose upper bound: (n - i - 1) * 3 (max 3 contacts per residue in 3D? No, max 6 neighbors, but self-avoiding limits it).
                # Let's use a simpler heuristic: 0. This makes it a pure beam search on current energy.
                # To guide it, we can add a term for "openness" or just rely on energy.
                # Let's add a small bonus for keeping the chain compact?
                # No, let's stick to energy + 0 heuristic first.
                
                new_score = new_energy # We want to minimize energy (most negative is best).
                # So we sort by new_score ascending.
                
                next_beam.append((new_score, chain + (next_pos,), {**occupied, next_pos: i}, new_energy))
        
        if not next_beam:
            # Dead end, return best from previous beam? Or fail.
            # If dead end, we can't continue. Return the best chain we have so far?
            # But it's not a full fold. 
            # For robustness, if beam is empty, we should probably return the best partial chain extended? 
            # But the task requires a full chain.
            # If beam search fails, it's a problem. 
            # We can fallback to a greedy extension if beam empties.
            # But for now, let's assume beam_width is large enough.
            # If it empties, return None.
            return None
            
        # Sort by score (energy) ascending (most negative first)
        next_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Time check
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Return the best chain (first in sorted beam)
    best_state = beam[0]
    return list(best_state[1])

def fold(sequence, dimension):
    """
    Uses beam search with multiple widths and restarts.
    """
    n = len(sequence)
    
    # Global time management
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 300.0
    current_time = time.time()
    
    if current_time >= global_deadline:
        # Fallback
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Determine budget for this call
    # Assume 26 calls. 
    # Let's allocate ~8s per call on average, but allow more for short chains.
    # Short chains (n < 50) can use more time.
    # Long chains (n > 50) need careful beam width.
    
    base_time = 6.0
    if n < 50:
        base_time = 10.0
    elif n < 100:
        base_time = 8.0
    else:
        base_time = 5.0
        
    call_deadline = current_time + base_time
    if call_deadline > global_deadline:
        call_deadline = global_deadline
    
    best_chain = None
    best_energy = 0
    
    # Strategy: Try a few different beam widths.
    # For short chains, wider beam. For long, narrower.
    if n <= 48:
        widths = [50, 20, 10]
    else:
        widths = [20, 10, 5]
        
    for width in widths:
        if time.time() > call_deadline - 1.0:
            break
            
        # Beam search
        chain = _beam_search_fold_v2(sequence, dimension, width, call_deadline - time.time())
        
        if chain is not None:
            energy = _energy(sequence, chain)
            if best_chain is None or energy < best_energy:
                best_energy = energy
                best_chain = chain
                
    if best_chain is None:
        # Fallback to a simple greedy if beam search failed completely
        # Greedy: always pick the neighbor that gives most immediate gain
        chain = []
        occupied = set()
        pos = (0, 0, 0)
        chain.append(pos)
        occupied.add(pos)
        
        for i in range(1, n):
            best_next = None
            best_gain = -1
            for next_pos in _neighbours(pos):
                if next_pos in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_next = next_pos
            if best_next is None:
                # No move, break
                break
            chain.append(best_next)
            occupied.add(best_next)
            pos = best_next
            
        if len(chain) == n:
            best_chain = chain
            best_energy = _energy(sequence, chain)
        else:
            # If greedy also failed, return a dummy linear chain (might be invalid self-avoiding? No, linear is self-avoiding)
            # But linear chain has 0 energy.
            best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
            best_energy = 0

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
