import time
import random

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, i, q):
    """Calculate new H-H contacts when placing residue i at position q."""
    if sequence[i] != 'H':
        return 0
    x, y, z = q
    delta = 0
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        r = (x + dx, y + dy, z + dz)
        if r in occupied:
            j = occupied[r]
            if j != i - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _greedy_complete(sequence, chain, occupied, n, deadline):
    """Greedily complete a partial chain. Returns None if stuck."""
    for j in range(len(chain), n):
        if time.time() > deadline:
            return None
        last_pos = chain[-1]
        valid_mvs = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in occupied:
                valid_mvs.append(q)
        if not valid_mvs:
            return None
        # Pick move maximizing H-contacts, then free space
        best_q = None
        best_score = -1
        for q in valid_mvs:
            score = _calc_delta_energy(sequence, occupied, j, q)
            free = 0
            x, y, z = q
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (x + dx, y + dy, z + dz)
                if r not in occupied:
                    free += 1
            score += free * 0.1
            if score > best_score:
                best_score = score
                best_q = q
        if best_q is None:
            return None
        chain.append(best_q)
        occupied[best_q] = j
    return chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    start_time = time.time()
    # Use most of the 240s budget; leave 15s safety margin
    deadline = start_time + 225.0
    
    # Beam width: larger = better quality but slower per step
    # For n up to 136, width 80 should be manageable within time budget
    beam_width = 80
    
    # Initial state
    init_chain = [(0, 0, 0)]
    init_occupied = {(0, 0, 0): 0}
    init_energy = 0
    
    beam = [(init_energy, init_chain, init_occupied)]
    
    best_complete_chain = None
    best_complete_energy = 0
    
    for i in range(1, n):
        new_beam = []
        
        for energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    delta = _calc_delta_energy(sequence, occupied, i, q)
                    new_energy = energy - delta
                    
                    new_chain = chain + [q]
                    new_occupied = occupied.copy()
                    new_occupied[q] = i
                    
                    new_beam.append((new_energy, new_chain, new_occupied))
        
        if not new_beam:
            break
            
        # Sort by energy (ascending = most negative = best)
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Time check: if we're running low, try to complete the best partial
        if time.time() > deadline - 5.0:
            # Try to greedily complete the best partial chain
            best_energy, best_chain, best_occupied = beam[0]
            completed = _greedy_complete(sequence, best_chain.copy(), best_occupied.copy(), n, deadline)
            if completed is not None and len(completed) == n:
                # Compute final energy
                occ = {}
                for idx, pos in enumerate(completed):
                    occ[pos] = idx
                contacts = 0
                for idx, pos in enumerate(completed):
                    if sequence[idx] == 'H':
                        x, y, z = pos
                        for dx, dy, dz in _NEIGHBOR_OFFSETS:
                            r = (x + dx, y + dy, z + dz)
                            if r in occ:
                                j = occ[r]
                                if j != idx - 1 and j != idx + 1 and sequence[j] == 'H':
                                    contacts += 1
                final_energy = -contacts // 2  # Each contact counted twice
                if best_complete_chain is None or final_energy < best_complete_energy:
                    best_complete_chain = completed
                    best_complete_energy = final_energy
            break
        
        # If this is the last step, we have complete chains
        if i == n - 1:
            best_energy, best_chain, _ = beam[0]
            # Compute actual energy to verify
            occ = {}
            for idx, pos in enumerate(best_chain):
                occ[pos] = idx
            contacts = 0
            for idx, pos in enumerate(best_chain):
                if sequence[idx] == 'H':
                    x, y, z = pos
                    for dx, dy, dz in _NEIGHBOR_OFFSETS:
                        r = (x + dx, y + dy, z + dz)
                        if r in occ:
                            j = occ[r]
                            if j != idx - 1 and j != idx + 1 and sequence[j] == 'H':
                                contacts += 1
            final_energy = -contacts // 2
            return [list(p) for p in best_chain]
    
    # If we broke out early or beam became empty, try to complete best partial
    if beam:
        best_energy, best_chain, best_occupied = beam[0]
        completed = _greedy_complete(sequence, best_chain.copy(), best_occupied.copy(), n, deadline)
        if completed is not None and len(completed) == n:
            return [list(p) for p in completed]
    
    # Fallback: straight line
    return [[i, 0, 0] for i in range(n)]
