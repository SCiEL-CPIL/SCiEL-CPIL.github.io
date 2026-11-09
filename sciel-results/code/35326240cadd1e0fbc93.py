"""
HP lattice protein folding on the cubic lattice.
Strategy: Tail Regrowth with Simulated Annealing.
Instead of recombining chains (which creates invalid structures requiring repair),
we maintain a valid self-avoiding walk and iteratively delete a random suffix
(the tail) and regrow it using a temperature-controlled Monte Carlo walk.
This allows the chain to explore large regions of the configuration space
by "unfolding" and "refolding" parts of the protein, effectively performing
a global search while maintaining validity at every step.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin before 330s kill

# Precompute neighbors for speed? No, coordinates vary.
# But we can optimize the neighbor generation.

def _neighbors(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, z+1), (x, z-1)]

def _energy(sequence, chain):
    """
    Compute total energy: negative count of non-adjacent H-H contacts.
    Uses a dictionary for O(1) lookup of residues at positions.
    """
    if not chain:
        return 0
    
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbors(p):
                j = where.get(q)
                # j is not None means q is occupied by residue j
                # We need j > i + 1 to avoid counting the bond itself (i, i+1)
                # and to avoid double counting (we only count if j > i)
                # The condition j > i + 1 ensures we don't count the adjacent bond.
                # Since we iterate i from 0 to n-1, and only count if j > i, 
                # the condition j > i + 1 is sufficient for "non-adjacent along chain".
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, bias=1.0):
    """
    Fast initial self-avoiding walk with heuristic bias towards H contacts.
    """
    n = len(sequence)
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = _neighbors(last)
        
        options = []
        for site in neighbors:
            if site in occupied:
                continue
            
            sx, sy, sz = site
            h_score = 0
            free_score = 0
            
            site_neighbors = _neighbors(site)
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
                else:
                    free_score += 1
            
            score = h_score * bias + free_score * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _regrow_tail(sequence, chain, cut_index, T, rng):
    """
    Delete chain[cut_index:] and regrow it using Monte Carlo with temperature T.
    Returns the new chain and the energy change.
    """
    n = len(sequence)
    if cut_index >= n:
        return chain, 0
    
    # 1. Cut the tail
    prefix = chain[:cut_index]
    # The tail starts at index cut_index.
    # We need to remove the tail from the occupied set.
    occupied = {p: i for i, p in enumerate(prefix)}
    
    new_chain = prefix[:]
    last_pos = prefix[-1]
    
    # 2. Regrow from cut_index to n-1
    for i in range(cut_index, n):
        neighbors = _neighbors(last_pos)
        
        # Filter valid moves
        valid_moves = []
        for site in neighbors:
            if site in occupied:
                continue
            
            # Calculate heuristic score for this move
            # 1. H-contact potential
            h_score = 0
            site_neighbors = _neighbors(site)
            for sn in site_neighbors:
                if sn in occupied:
                    idx = occupied[sn]
                    if sequence[idx] == 'H' and abs(idx - i) > 1:
                        h_score += 1
            
            # 2. Free space (to avoid trapping)
            free_score = 0
            for sn in site_neighbors:
                if sn not in occupied:
                    free_score += 1
            
            # Combine scores
            # We want to prefer moves that create H contacts and keep options open.
            # The temperature T controls the randomness.
            # High T -> more random. Low T -> more greedy.
            # We can use a Boltzmann-like distribution or simple weighted random.
            
            # Let's define a "cost" or "energy delta" approximation for the move.
            # If we place H at site, and there are k H neighbors (non-adjacent), 
            # the energy delta is -k.
            # If we place P, energy delta is 0.
            
            delta_e = 0
            if sequence[i] == 'H':
                delta_e = -h_score
            
            # We want to minimize energy (maximize h_score for H).
            # So we want to accept moves with lower delta_e.
            # But we also need to avoid trapping.
            # Simple heuristic: score = -delta_e * bias_h + free_score * bias_f
            # Let's use a soft potential.
            
            # For simplicity and speed, let's use a weighted random selection
            # where weight = exp(-delta_e / T) * (1 + 0.1 * free_score)
            # Note: delta_e is negative for good moves, so exp(-delta_e/T) > 1.
            
            weight = math.exp(-delta_e / T) * (1.0 + 0.2 * free_score)
            valid_moves.append((weight, site))
            
        if not valid_moves:
            # Trapped! This regrowth attempt failed.
            # Return the original chain (effectively undoing the cut)
            # Or we could try a different cut index in the main loop.
            # For now, return the original chain to signal failure.
            return chain, 0
        
        # Normalize weights for random selection
        total_weight = sum(w for w, _ in valid_moves)
        r = rng.random() * total_weight
        cumulative = 0
        chosen_site = valid_moves[-1][1]
        for w, site in valid_moves:
            cumulative += w
            if r <= cumulative:
                chosen_site = site
                break
        
        # Move to chosen_site
        new_chain.append(chosen_site)
        occupied[chosen_site] = i
        last_pos = chosen_site
        
    return new_chain, 0

def _fold_sequence(sequence, dimension, time_budget, rng):
    n = len(sequence)
    
    # 1. Initialize
    initial_chain = None
    for _ in range(10):
        initial_chain = _greedy_grow(sequence, dimension, rng, bias=2.0)
        if initial_chain:
            break
    
    if not initial_chain:
        # Fallback: straight line (might be invalid if n > 1 and dimension constraints? No, straight line is always valid SAW)
        # But it's high energy.
        return [(i, 0, 0) for i in range(n)]
    
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # 2. Tail Regrowth SA Loop
    # Start with high temperature, cool down.
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Choose a cut index
        # Cut closer to the end for fine-tuning, further for global moves.
        # Randomly choose cut index between n//4 and n-1.
        # If n is small, adjust.
        min_cut = max(1, n // 4)
        max_cut = n - 1
        
        if min_cut > max_cut:
            min_cut = 1
            max_cut = 1
            
        cut_idx = rng.randint(min_cut, max_cut)
        
        # Regrow
        new_chain, _ = _regrow_tail(sequence, current_chain, cut_idx, T, rng)
        
        # If regrow failed (returned original), continue to next step
        if new_chain is current_chain:
            # Try a smaller tail or different T? Just continue.
            # But we should check if we are stuck.
            pass
        else:
            new_energy = _energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
            
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Check time
        if time.time() - start_time > time_budget:
            break
            
    return best_chain

def fold(sequence, dimension):
    """
    Best conformation using Tail Regrowth with Simulated Annealing.
    """
    n = len(sequence)
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 1.0:
        # Emergency fallback
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time per sequence
    # 26 sequences. 325s total.
    # Let's use a fixed 10s per sequence, capped by remaining.
    budget = 10.0
    if remaining - budget < 5.0:
        budget = remaining - 5.0
        
    if budget < 1.0:
        budget = 0.5

    rng = random.Random(54321 + len(sequence))
    
    best_chain = _fold_sequence(sequence, dimension, budget, rng)
    
    # Final check: ensure self-avoidance
    if len(set(best_chain)) != n:
        # Fallback
        return [[i, 0, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
