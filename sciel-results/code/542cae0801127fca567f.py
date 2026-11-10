"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin
_PER_SEQ_BUDGET = _TOTAL_BUDGET / 26.0 * 0.92  # ~10.5s per sequence

# Precomputed neighbor offsets for 3D lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D lattice."""
    for offset in _NEIGHBOR_OFFSETS:
        yield (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2])

def _energy(sequence, chain):
    """
    Calculate the energy of the chain.
    Energy = - (number of H-H contacts on non-adjacent lattice neighbors).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_grow(sequence, dimension, rng, beam_width=100):
    """
    Beam search growth.
    Keeps the top `beam_width` partial chains at each step, scoring them by
    current H-H contacts and a small penalty for low free space to avoid dead ends.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Each beam entry: (score, chain_list, occupied_set)
    # We use lists for chains for easier appending, and sets for occupied for O(1) lookup
    # To save memory, we could use tuples, but sets are faster for growth.
    # However, copying sets is expensive.
    # Alternative: Use a list for chain, and a set for occupied.
    
    # Initial state
    start = (0, 0, 0)
    initial_chain = [start]
    initial_occ = {start}
    initial_score = 0.0
    
    beams = [(initial_score, initial_chain, initial_occ)]
    
    for i in range(1, n):
        new_beams = []
        residue = sequence[i]
        
        for score, chain, occ in beams:
            last = chain[-1]
            
            # Expand from last position
            for offset in _NEIGHBOR_OFFSETS:
                new_pos = (last[0] + offset[0], last[1] + offset[1], last[2] + offset[2])
                
                if new_pos in occ:
                    continue
                
                # Calculate gain
                gain = 0
                if residue == "H":
                    for offset2 in _NEIGHBOR_OFFSETS:
                        neighbor_pos = (new_pos[0] + offset2[0], new_pos[1] + offset2[1], new_pos[2] + offset2[2])
                        if neighbor_pos in occ:
                            # Check if the occupant is H and not the previous residue (which is i-1)
                            # Since we only store positions in occ, we need to map position to index?
                            # Wait, `occ` is a set of positions. We don't know which index is at which position easily without a dict.
                            # Using a dict `pos_to_idx` is better for energy calculation, but sets are faster for collision.
                            # Let's use a dict for occupied to allow checking residue type.
                            # This increases memory but is necessary for correct contact counting during growth.
                            pass 
                
                # The simple set approach above doesn't allow checking if the neighbor is H.
                # We need to know the residue type of the neighbor.
                # So `occ` must be a dict {pos: index}.
                
                # Let's restart the beam structure with dicts.
                # It's slower but correct.
                pass
        
        # Since the simple set approach fails for H-checking, we must use dicts.
        # Let's implement it properly.
        # This might be slow. Let's optimize.
        # Actually, for beam search, we can just store the chain and calculate energy on the fly? No, that's O(N^2).
        # We need incremental energy.
        # So we need to know if a neighbor is H.
        # So `occ` must be {pos: residue_char} or {pos: index}.
        
        # Let's try a different approach for beam search to be faster:
        # Store (score, chain_tuple, occupied_dict)
        # occupied_dict: {pos: index}
        
        # Re-initialize
        beams = [(0.0, (start,), {start: 0})]
        
        for i in range(1, n):
            new_beams = []
            residue = sequence[i]
            
            for score, chain_tuple, occ_dict in beams:
                last = chain_tuple[-1]
                
                for offset in _NEIGHBOR_OFFSETS:
                    new_pos = (last[0] + offset[0], last[1] + offset[1], last[2] + offset[2])
                    
                    if new_pos in occ_dict:
                        continue
                    
                    # Calculate gain
                    gain = 0
                    if residue == "H":
                        for offset2 in _NEIGHBOR_OFFSETS:
                            neighbor_pos = (new_pos[0] + offset2[0], new_pos[1] + offset2[1], new_pos[2] + offset2[2])
                            if neighbor_pos in occ_dict:
                                neighbor_idx = occ_dict[neighbor_pos]
                                if neighbor_idx < i - 1 and sequence[neighbor_idx] == "H":
                                    gain += 1
                    
                    # Free space penalty (optional, to avoid dead ends)
                    # Count free neighbors of new_pos
                    free_count = 0
                    for offset3 in _NEIGHBOR_OFFSETS:
                        check_pos = (new_pos[0] + offset3[0], new_pos[1] + offset3[1], new_pos[2] + offset3[2])
                        if check_pos not in occ_dict and check_pos != last: # last is occupied by i-1
                            free_count += 1
                    
                    # Score: prioritize contacts, slight preference for free space
                    new_score = score + gain * 10.0 + free_count * 0.1 + rng.random() * 0.5
                    
                    new_chain = chain_tuple + (new_pos,)
                    new_occ = dict(occ_dict)
                    new_occ[new_pos] = i
                    
                    new_beams.append((new_score, new_chain, new_occ))
            
            # Prune to beam width
            if len(new_beams) > beam_width:
                new_beams.sort(key=lambda x: x[0], reverse=True)
                new_beams = new_beams[:beam_width]
            
            beams = new_beams
            
            if not beams:
                break
        
        if not beams:
            return None
        
        # Return the best beam
        best_beam = max(beams, key=lambda x: x[0])
        return list(best_beam[1])

def _local_search(sequence, chain, rng, deadline):
    """
    Hill-climbing local search using single-residue moves.
    """
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    occupied = {p: i for i, p in enumerate(best)}
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(1, n - 1))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() >= deadline:
                break
                
            old_pos = best[i]
            prev_pos = best[i - 1]
            next_pos = best[i + 1]
            
            candidates = []
            for site in _neighbours(prev_pos):
                if site == old_pos:
                    continue
                if sum(abs(site[d] - next_pos[d]) for d in range(3)) == 1:
                    if site not in occupied:
                        candidates.append(site)
            
            if not candidates:
                continue
            
            for site in candidates:
                delta_contacts = 0
                
                if is_h[i]:
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts -= 1
                    
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i:
                            if is_h[j]:
                                delta_contacts += 1
                
                if delta_contacts > 0:
                    del occupied[old_pos]
                    occupied[site] = i
                    best[i] = site
                    best_e -= delta_contacts
                    improved = True
                    break
    
    return best, best_e

def fold(sequence, dimension):
    """
    Best conformation using beam search initialization and local search refinement.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    current_time = time.time()
    elapsed_global = current_time - _TOTAL_START
    remaining_global = _TOTAL_BUDGET - elapsed_global
    
    seq_deadline = min(current_time + _PER_SEQ_BUDGET, _TOTAL_START + _TOTAL_BUDGET)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam Search Initialization
    # Use a few different beam widths or just one good one?
    # One good beam search is usually better than many greedy.
    # Let's try 1 beam search with width 50-100.
    # And maybe a few greedy ones as fallback/diversity if time permits.
    
    beam_width = 100 if n <= 48 else 50
    
    # Try beam search
    if time.time() < seq_deadline - 2.0:
        chain = _beam_grow(sequence, dimension, rng, beam_width)
        if chain is not None:
            value = _energy(sequence, chain)
            if value < best_energy:
                best_chain = chain
                best_energy = value
    
    # If beam search failed or didn't find anything, try a few greedy growths
    if best_chain is None or time.time() < seq_deadline - 2.0:
        # Quick greedy fallback
        for restart in range(10):
            if time.time() >= seq_deadline - 2.0:
                break
            # Simple greedy
            start = (0, 0, 0)
            chain = [start]
            occupied = {start: 0}
            valid = True
            for i in range(1, n):
                options = []
                last = chain[-1]
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    gain = 0
                    if sequence[i] == "H":
                        for q in _neighbours(site):
                            j = occupied.get(q)
                            if j is not None and j < i - 1 and sequence[j] == "H":
                                gain += 1
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
                    options.append((score, site))
                if not options:
                    valid = False
                    break
                best_score, site = max(options)
                occupied[site] = i
                chain.append(site)
            
            if valid:
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_chain = chain
                    best_energy = value
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local Search
    if time.time() < seq_deadline - 0.5:
        best_chain, best_energy = _local_search(sequence, best_chain, rng, seq_deadline - 0.5)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
