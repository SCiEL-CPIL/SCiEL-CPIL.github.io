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
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
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
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _beam_search(sequence, rng, beam_width=50):
    """
    Beam search: maintain top-K partial chains, scoring by contacts + heuristic.
    Returns best complete chain found.
    """
    n = len(sequence)
    if n == 0:
        return None
    
    is_h = [c == 'H' for c in sequence]
    num_h = sum(is_h)
    
    # Initial state: start at origin
    # State: (chain_list, occupied_dict, contacts_so_far)
    beam = [([(0, 0)], {(0, 0): 0}, 0)]
    
    for i in range(1, n):
        new_beam = []
        
        for chain, occupied, contacts in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Count new H-H contacts this placement makes
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                
                new_contacts = contacts + gain
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                # Heuristic: estimate remaining potential contacts
                # Simple: number of unplaced H residues * 0.5 (conservative)
                remaining_h = sum(is_h[i+1:])
                heuristic = remaining_h * 0.5
                
                # Score: maximize contacts + heuristic
                score = new_contacts + heuristic
                
                new_beam.append((score, new_chain, new_occupied, new_contacts))
        
        if not new_beam:
            break
        
        # Keep top beam_width states
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = [(chain, occ, cont) for _, chain, occ, cont in new_beam[:beam_width]]
    
    if not beam:
        return None
    
    # Return the best complete chain
    best = max(beam, key=lambda x: x[2])
    return best[0]

def _local_search(sequence, chain, rng, max_steps=300):
    """
    Refine using kink moves with simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, chain)
    current_energy = -current_contacts
    is_h = [c == 'H' for c in sequence]
    
    start_temp = 1.5
    end_temp = 0.02
    
    for step in range(max_steps):
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Kink move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
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
            chain[i] = curr
            del occupied[new_pos]
            occupied[curr] = i
                
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence using beam search followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Adaptive beam width based on sequence length and time
    # Longer sequences need smaller beams to fit in time
    base_beam = max(20, min(100, 2000 // max(1, n)))
    
    max_iterations = 300
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Adaptive beam width: start wider, narrow if time is tight
        time_frac = _time_remaining() / 110.0
        if time_frac > 0.7:
            beam_w = base_beam
        elif time_frac > 0.4:
            beam_w = max(10, base_beam // 2)
        else:
            beam_w = max(5, base_beam // 4)
        
        chain = _beam_search(sequence, rng, beam_width=beam_w)
        if chain is None:
            continue
        
        # Local search refinement
        ls_steps = max(50, min(500, 10000 // max(1, n)))
        chain, energy = _local_search(sequence, chain, rng, max_steps=ls_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
