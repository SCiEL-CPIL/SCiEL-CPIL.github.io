# EVOLVE-BLOCK-START
import time
import sys
import math
import heapq
import random

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_fold(sequence, dimension):
    """
    Fast greedy fallback to ensure we always return a valid fold.
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    chain = [(0,)*dimension]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        last_pos = chain[-1]
        best_next = None
        best_gain = -1
        best_free = -1
        
        for next_pos in _neighbours(last_pos):
            if next_pos in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(next_pos):
                    if q in occupied:
                        idx = occupied[q]
                        if sequence[idx] == "H":
                            gain += 1
            
            free = sum(1 for q in _neighbours(next_pos) if q not in occupied)
            
            # Prioritize gain, then openness
            if gain > best_gain or (gain == best_gain and free > best_free):
                best_gain = gain
                best_free = free
                best_next = next_pos
        
        if best_next is None:
            # Should not happen for reasonable n, but if it does, stop
            break
            
        chain.append(best_next)
        occupied[best_next] = i
        
    # If we didn't place all residues, we have a problem. 
    # For the fallback, we just return what we have? No, must be full length.
    # If greedy fails, we return a linear chain (which is always valid)
    if len(chain) < n:
        return [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    return chain

def _astar_fold(sequence, dimension, time_limit):
    """
    A* search with admissible heuristic.
    State: (f, g, h, chain_tuple, occupied_dict)
    f = g + h
    g = current energy (negative)
    h = heuristic estimate of future energy (negative or zero)
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start_time = time.time()
    start_pos = (0,)*dimension
    initial_occupied = {start_pos: 0}
    initial_chain = (start_pos,)
    initial_energy = 0
    
    # Heuristic: 
    # The maximum possible additional energy is 0 (if no more contacts are made).
    # A better admissible heuristic: 
    # For each remaining H residue, it can make at most 6 contacts, but self-avoidance limits this.
    # A safe admissible heuristic is 0, because energy can only decrease (become more negative) or stay same.
    # Wait, we want to MINIMIZE energy.
    # g is the cost so far. Since we minimize energy, g = -contacts_so_far.
    # h should be a lower bound on the remaining cost (i.e., a value >= true remaining cost).
    # Since remaining cost is <= 0 (energy goes down), a lower bound is -infinity? No.
    # Let's reframe: we want to find the path with the most negative total energy.
    # Equivalent to maximizing total contacts.
    # Let's maximize contacts.
    # g = contacts_so_far
    # h = upper_bound_on_future_contacts
    # f = g + h
    # We want to maximize f. So we use a max-heap.
    
    # Upper bound on future contacts:
    # Each remaining H residue can form at most 6 contacts.
    # But a contact involves 2 residues.
    # A simple admissible heuristic: 
    # Count remaining H residues. Each can contribute at most 6 contacts? 
    # Actually, a contact is between two H's. 
    # If we have k remaining H's, the max contacts involving them is k * (k-1) / 2 + k * (H's already placed).
    # This is complex to compute dynamically.
    # A simpler, safe upper bound: 
    # The total number of H-H contacts in the final chain cannot exceed the number of H-H pairs in the sequence.
    # But that's for the whole sequence.
    # For the remaining part, the max contacts is bounded by the number of H's remaining * 6? 
    # Let's use a very simple admissible heuristic: 
    # h = 0. This makes it Dijkstra's (or uniform cost search) if we just expand by g.
    # But we want to guide it.
    # Let's try h = number of remaining H residues. This is an upper bound on the number of new contacts?
    # No, one H can make multiple contacts.
    # Let's stick with h = 0 for now to ensure correctness, but use a priority queue to explore best g first.
    # This is effectively a branch-and-bound where we prune if current g is worse than best found.
    
    # Actually, let's use a simple heuristic: 
    # h = 0. 
    # We will use a max-heap on (g, chain).
    # We will keep track of the best energy found for a given (chain_state) to avoid revisiting.
    # State is defined by the set of occupied points and the last point? 
    # No, the chain order matters for energy calculation? 
    # Energy is defined by the set of points and the sequence. 
    # So two different chains visiting the same set of points in different orders might have different energies?
    # No, energy depends on which residues are at which points.
    # So the state must include the mapping of index -> point.
    # This makes the state space huge.
    
    # Given the time limit, A* with full state space is likely too slow for n>20.
    # Let's use a beam search but with a better heuristic and tie-breaking.
    
    # Let's stick to the beam search structure but improve the heuristic.
    # Heuristic: 
    # For each remaining H residue, estimate the max contacts it can make with already placed H's.
    # This is easy to compute: for the current last H, check neighbors.
    # But for future H's, we don't know where they will be.
    
    # Let's use a simple "density" heuristic: 
    # Encourage compactness.
    # h = -sum(distance_from_centroid) for remaining residues?
    # This is not admissible for energy.
    
    # Let's try a different approach: 
    # Use the beam search from the previous program but with a much wider beam for short chains
    # and a local search step at the end.
    
    return None # Placeholder, we will use the improved beam search below

def _beam_search_fold_v3(sequence, dimension, beam_width, time_limit):
    """
    Improved beam search with better heuristic and local search.
    """
    n = len(sequence)
    if n <= 1:
        return [tuple([0]*dimension)]
    
    start_time = time.time()
    start_pos = (0,)*dimension
    
    # State: (score, chain_tuple, occupied_dict, energy)
    # score = energy + heuristic
    # We want to maximize score (less negative energy is better? No, energy is negative, so more negative is better).
    # Let's minimize energy.
    # score = energy. We want to find the minimum energy.
    # So we sort by energy ascending.
    
    initial_occupied = {start_pos: 0}
    initial_chain = (start_pos,)
    initial_energy = 0
    
    # Heuristic: 
    # A simple heuristic that encourages compactness: 
    # h = -0.1 * sum of distances from origin for all placed residues?
    # This is not admissible, but might guide the search.
    # Let's try h = 0 first, but use a larger beam.
    
    beam = [(0, initial_chain, initial_occupied, initial_energy)] # score, chain, occupied, energy
    
    for i in range(1, n):
        next_beam = []
        res = sequence[i]
        
        for score, chain, occupied, energy in beam:
            last_pos = chain[-1]
            
            for next_pos in _neighbours(last_pos):
                if next_pos in occupied:
                    continue
                
                gain = 0
                if res == "H":
                    for q in _neighbours(next_pos):
                        if q in occupied:
                            idx = occupied[q]
                            if sequence[idx] == "H":
                                gain += 1
                
                new_energy = energy - gain
                
                # Heuristic: 
                # Let's add a small penalty for spreading out too much.
                # Calculate bounding box size?
                # Or just distance from origin?
                # Let's try: h = -0.05 * (abs(next_pos[0]) + abs(next_pos[1]) + abs(next_pos[2]))
                # This encourages staying near the origin.
                h = -0.05 * (abs(next_pos[0]) + abs(next_pos[1]) + abs(next_pos[2]))
                
                new_score = new_energy + h
                
                next_beam.append((new_score, chain + (next_pos,), {**occupied, next_pos: i}, new_energy))
        
        if not next_beam:
            return None
            
        # Sort by score (energy + heuristic) ascending
        next_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Time check
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Return the best chain
    # We should pick the one with the lowest actual energy, not just score.
    best_state = min(beam, key=lambda x: x[3])
    return list(best_state[1])

def _local_optimize(sequence, chain, time_limit):
    """
    Simple local search: try to move a single residue to a nearby empty spot if it improves energy.
    """
    n = len(sequence)
    if n <= 1:
        return chain
    
    start_time = time.time()
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(1, n-1): # Don't move ends
            p = current_chain[i]
            for q in _neighbours(p):
                if q in occupied:
                    continue
                # Try moving i to q
                # Check if this move is valid (self-avoidance is maintained by q not in occupied)
                # Calculate new energy
                # This is expensive to do naively.
                # Let's just swap in the chain and recompute energy? 
                # For n=48, recomputing energy is O(n^2) or O(n*6).
                # Let's try it.
                
                # Save old state
                old_p = current_chain[i]
                del occupied[old_p]
                current_chain[i] = q
                occupied[q] = i
                
                new_energy = _energy(sequence, current_chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert
                    current_chain[i] = old_p
                    del occupied[q]
                    occupied[old_p] = i
                    
    return current_chain

def fold(sequence, dimension):
    """
    Uses beam search with local optimization.
    """
    n = len(sequence)
    
    # Global time management
    if not hasattr(fold, '_global_start'):
        fold._global_start = time.time()
        
    global_deadline = fold._global_start + 300.0
    current_time = time.time()
    
    if current_time >= global_deadline:
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Determine budget for this call
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
    
    # Strategy: Beam search with local optimization
    if n <= 48:
        widths = [100, 50, 20]
    else:
        widths = [50, 20, 10]
        
    for width in widths:
        if time.time() > call_deadline - 2.0:
            break
            
        # Beam search
        chain = _beam_search_fold_v3(sequence, dimension, width, call_deadline - time.time() - 1.0)
        
        if chain is not None and len(chain) == n:
            # Local optimization
            opt_chain = _local_optimize(sequence, chain, 1.0)
            energy = _energy(sequence, opt_chain)
            if best_chain is None or energy < best_energy:
                best_energy = energy
                best_chain = opt_chain
                
    if best_chain is None:
        # Fallback to greedy
        best_chain = _greedy_fold(sequence, dimension)
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
