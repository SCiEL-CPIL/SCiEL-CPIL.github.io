"""HP lattice protein folding with beam search and local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU. Safety margin to 110s.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    """Calculate total H-H contacts for a chain."""
    n = len(chain)
    if n == 0:
        return 0
    
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        # Check 4 neighbors
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _beam_search(sequence, beam_width=50, rng=None):
    """
    Beam search for HP lattice protein folding.
    Maintains multiple partial conformations, scoring by contact count.
    Returns list of (chain, contacts) for all complete conformations.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    is_h = [c == 'H' for c in sequence]
    
    # Each beam entry: (chain, occupied_dict, contacts_so_far)
    # Start with one beam entry at origin
    start = (0, 0)
    beam = [([start], {start: 0}, 0)]
    
    for i in range(1, n):
        if _time_remaining() < 2.0:
            break
        
        # Expand all beam entries
        candidates = []
        
        for chain, occupied, contacts in beam:
            current = chain[-1]
            
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Count new H-H contacts this placement would make
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                # Count free neighbours for flexibility
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                # Score: contacts dominate, then free space
                score = (contacts + gain) * 100.0 + free * 0.1
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_contacts = contacts + gain
                
                candidates.append((score, new_chain, new_occupied, new_contacts))
        
        if not candidates:
            break
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(c, o, ct) for _, c, o, ct in candidates[:beam_width]]
        
        # If beam is empty, no valid conformations
        if not beam:
            break
    
    # Return all complete conformations from the beam
    if len(beam[0][0]) == n:
        return [(chain, contacts) for chain, occupied, contacts in beam]
    else:
        return []

def _local_search(sequence, chain, rng, max_steps=1000):
    """
    Refine the chain using kink moves with simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain)
    
    # Precompute occupied set and energy
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, chain)
    current_energy = -current_contacts
    is_h = [c == 'H' for c in sequence]
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.05
    
    for step in range(max_steps):
        if _time_remaining() < 1.0:
            break
        
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Kink move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid new positions for curr
        candidates = set()
        for p in _neighbours(prev):
            if p == curr:
                continue
            dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
            if dist == 1:
                candidates.add(p)
        
        if not candidates:
            continue
            
        new_pos = rng.choice(list(candidates))
        
        if new_pos in occupied:
            continue
        
        # Apply move
        del occupied[curr]
        occupied[new_pos] = i
        chain[i] = new_pos
        
        new_contacts = _count_contacts(sequence, chain)
        new_energy = -new_contacts
        delta = new_energy - current_energy
        
        if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
            current_energy = new_energy
            current_contacts = new_contacts
        else:
            # Revert
            chain[i] = curr
            del occupied[new_pos]
            occupied[curr] = i
            
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we look for lower (more negative) values
    
    # Perform beam search with different beam widths
    beam_widths = [20, 50, 100]
    
    for bw in beam_widths:
        if _time_remaining() < 5.0:
            break
        
        # Run beam search
        results = _beam_search(sequence, beam_width=bw, rng=rng)
        
        if not results:
            continue
        
        # Local search on top results
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by contacts (higher is better)
        
        for chain, contacts in results[:5]:  # Try top 5 from beam
            if _time_remaining() < 2.0:
                break
            
            refined_chain, energy = _local_search(sequence, chain, rng, max_steps=300)
            
            if energy < best_energy:
                best_energy = energy
                best_chain = refined_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
