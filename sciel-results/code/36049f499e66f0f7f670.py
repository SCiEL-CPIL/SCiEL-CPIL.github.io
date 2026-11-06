import time
import random

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _incremental_energy(sequence, chain, occupied, last_idx, new_pos):
    """
    Calculate the change in energy when adding new_pos at index last_idx.
    Assumes chain has last_idx elements (indices 0 to last_idx-1).
    new_pos is the position for index last_idx.
    """
    if sequence[last_idx] != 'H':
        return 0
    
    delta = 0
    x, y, z = new_pos
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        q = (x + dx, y + dy, z + dz)
        if q in occupied:
            j = occupied[q]
            # j is the index of the residue at q
            # We only count contacts with j < last_idx (previous residues)
            # And j != last_idx - 1 (adjacent in chain is not a contact)
            if j < last_idx - 1:
                if sequence[j] == 'H':
                    delta += 1
    return delta

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    max_time = 10.0 # Safety margin
    
    # Beam Search Parameters
    beam_width = 50
    
    # Initial state
    # Start at (0,0,0)
    init_chain = [(0, 0, 0)]
    init_occupied = {(0, 0, 0): 0}
    init_energy = 0
    
    beam = [(init_energy, init_chain, init_occupied)]
    
    rng = random.Random(42)
    
    for i in range(1, n):
        new_beam = []
        last_idx = i - 1
        
        for energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    # Calculate new energy
                    # We need to check neighbors of q for H-H contacts with previous H's
                    delta = 0
                    if sequence[i] == 'H':
                        for ndx, ndy, ndz in _NEIGHBOR_OFFSETS:
                            r = (q[0] + ndx, q[1] + ndy, q[2] + ndz)
                            if r in occupied:
                                j = occupied[r]
                                # j is index of residue at r
                                # Contact if j is H and j is not adjacent to i in chain
                                # Adjacent to i in chain is i-1.
                                # So if j != i-1, it's a contact.
                                if j != i - 1:
                                    if sequence[j] == 'H':
                                        delta += 1
                    
                    new_energy = energy - delta # Energy is negative contacts. More contacts = lower energy.
                    # Wait. Energy = -contacts.
                    # If we add 1 contact, energy decreases by 1.
                    # So new_energy = energy - 1.
                    # My delta is count of new contacts.
                    # So new_energy = energy - delta. Correct.
                    
                    new_chain = chain + [q]
                    new_occupied = occupied.copy()
                    new_occupied[q] = i
                    
                    new_beam.append((new_energy, new_chain, new_occupied))
        
        if not new_beam:
            # Dead end, should not happen with beam search if width is sufficient
            # Fallback: return best from previous beam? Or straight line?
            # If beam is empty, we lost all paths.
            # This is unlikely.
            break
            
        # Sort by energy (ascending, since we want min energy)
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        # To maintain diversity, we can keep some random ones if time permits
        # But strict top-k is usually fine for HP.
        beam = new_beam[:beam_width]
        
        # Time check
        if time.time() - start_time > max_time:
            # Return the best chain from the current beam
            # The beam contains partial chains. We need to complete them?
            # No, we should return the best *complete* chain.
            # If we break early, we don't have a complete chain.
            # Strategy: If time runs out, take the best partial chain and extend it greedily/randomly to complete it.
            
            best_energy, best_chain, best_occupied = beam[0]
            
            # Complete the chain
            for j in range(len(best_chain), n):
                last_pos = best_chain[-1]
                valid_mvs = []
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                    if q not in best_occupied:
                        valid_mvs.append(q)
                
                if not valid_mvs:
                    # Dead end. Try to backtrack? Too complex.
                    # Just append a dummy position? No, must be valid.
                    # If we hit a dead end, the chain is invalid.
                    # We should have avoided this by keeping the beam diverse.
                    # If this happens, return the best *valid* chain we can form.
                    # Since we can't form a complete one, we might fail.
                    # Let's try to pick a move that doesn't block future?
                    # For now, pick random.
                    pass
                
                if valid_mvs:
                    # Greedy: pick move with most H neighbors
                    best_q = None
                    best_score = -1
                    for q in valid_mvs:
                        score = 0
                        if sequence[j] == 'H':
                            for ndx, ndy, ndz in _NEIGHBOR_OFFSETS:
                                r = (q[0] + ndx, q[1] + ndy, q[2] + ndz)
                                if r in best_occupied:
                                    idx = best_occupied[r]
                                    if idx != j - 1 and sequence[idx] == 'H':
                                        score += 1
                        # Add free space
                        free = 0
                        for ndx, ndy, ndz in _NEIGHBOR_OFFSETS:
                            r = (q[0] + ndx, q[1] + ndy, q[2] + ndz)
                            if r not in best_occupied:
                                free += 1
                        score += free * 0.1
                        if score > best_score:
                            best_score = score
                            best_q = q
                    
                    if best_q:
                        best_chain.append(best_q)
                        best_occupied[best_q] = j
                    else:
                        # Should not happen
                        break
                else:
                    break
            
            if len(best_chain) == n:
                return [list(p) for p in best_chain]
            else:
                # Fallback to straight line if completion failed
                return [[i, 0, 0] for i in range(n)]

    # If we finished the loop, we have complete chains in the beam
    best_energy, best_chain, _ = beam[0]
    return [list(p) for p in best_chain]
