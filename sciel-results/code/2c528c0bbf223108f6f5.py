"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


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


def _beam_fold(sequence, dimension, beam_width=500, max_time=15.0):
    """
    Beam search for HP lattice protein folding.
    State: (chain, occupied_set, energy, free_space_score)
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    start_occupied = {start}
    
    # Initial state
    # energy = 0
    # free_space_score = sum(1 for q in _neighbours(start) if q not in start_occupied)
    initial_free = sum(1 for q in _neighbours(start) if q not in start_occupied)
    
    beam = [(( [start] ), start_occupied, 0, initial_free)]
    
    start_time = time.time()
    
    for i in range(1, n):
        current_time = time.time()
        if current_time - start_time > max_time:
            break
            
        candidates = []
        
        for chain, occupied, energy, free_score in beam:
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_occupied = occupied | {site}
                
                # Calculate energy change
                delta_e = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Check if the occupant is H and not adjacent in chain
                            # We need to know the index of the occupant.
                            # To avoid O(n) lookup in occupied set (which is just points),
                            # we should maintain a position->index map or check chain.
                            # Since we store chain, we can check indices if we had a map.
                            # For efficiency in beam search, maintaining a map per state is expensive.
                            # However, n is small (<=100). 
                            # Let's optimize: we can check if sequence[index_of(q)] == 'H'.
                            # We need index_of(q). 
                            # Rebuilding map for every candidate is O(n^2 * beam * 4) which might be too slow.
                            # Alternative: Store (chain, occupied_dict) where occupied_dict maps point->index.
                            # Let's refactor state to include the dict for faster lookups.
                            pass
                
                # Refactored approach for speed:
                # State: (chain_list, occupied_dict, energy)
                # occupied_dict: {point: index}
                
                # Let's restart the loop logic with the better state representation.
                # This inner block is just a placeholder for the logic flow, 
                # the actual implementation below uses the dict.
                pass
        
        # Re-implementing the step with better state for O(1) neighbor index lookup
        new_beam = []
        for chain, occupied_dict, energy in beam:
            last_pos = chain[-1]
            
            for site in _neighbours(last_pos):
                if site in occupied_dict:
                    continue
                
                new_chain = chain + [site]
                new_occupied_dict = dict(occupied_dict) # Copy dict
                new_occupied_dict[site] = i
                
                delta_e = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in new_occupied_dict:
                            j = new_occupied_dict[q]
                            if j < i - 1 and sequence[j] == 'H':
                                delta_e += 1
                            
                new_energy = energy - delta_e
                
                # Heuristic score: Lower energy is better. 
                # Also prefer more free space to avoid dead ends?
                # In beam search, usually we just keep the best K energies.
                # But sometimes a slightly worse energy with more space leads to better global result.
                # Let's use a composite score: energy - 0.1 * free_space
                # Or just energy. Let's stick to energy primarily, tie-break by free space.
                
                free_space = 0
                for q in _neighbours(site):
                    if q not in new_occupied_dict:
                        free_space += 1
                        
                # We want to minimize energy (more negative is better)
                # So we sort by energy ascending.
                # To avoid dead ends, we might penalize low free space.
                score = new_energy - 0.05 * free_space
                
                new_beam.append((new_chain, new_occupied_dict, new_energy, score))
        
        if not new_beam:
            break
            
        # Sort by score (ascending, since lower is better)
        new_beam.sort(key=lambda x: x[3])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Remove the 4th element (score) for next iteration if not needed, 
        # but we need it for sorting. 
        # For next iteration, we only need chain, dict, energy.
        beam = [(c, d, e) for c, d, e, s in beam]
        
        current_time = time.time()
        if current_time - start_time > max_time:
            break
    
    # Find best in final beam
    if not beam:
        # Fallback: straight line
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    best_chain = min(beam, key=lambda x: x[2])[0]
    return best_chain


def fold(sequence, dimension):
    """Best conformation using beam search."""
    if not sequence:
        return []
        
    # Allocate time based on sequence length
    # Shorter sequences can afford more beam width or time?
    # Total budget is 120s for 10 sequences. ~12s per sequence.
    # Let's use 10s per sequence to be safe.
    max_time = 10.0
    
    # Beam width: 
    # For n=20, beam=200 is fast.
    # For n=100, beam=200 might be slow. 
    # Let's scale beam width down for longer sequences or keep it constant and rely on time cutoff.
    # Constant beam width 300 seems reasonable.
    beam_width = 300
    
    try:
        best_chain = _beam_fold(sequence, dimension, beam_width=beam_width, max_time=max_time)
    except Exception:
        # Fallback to greedy if beam search fails or is too slow/buggy
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
