"""HP lattice protein folding with beam search and local refinement."""
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
    # Safety margin: evaluator kills at 210s, we want to finish by ~180s
    return 175.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _beam_grow(sequence, dimension, rng, beam_width=80):
    """
    Beam search chain growth: maintain beam_width partial chains, expanding each
    and keeping the top-scoring ones. Score favors H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Initial beam: start at origin
    beam = [([(0, 0)], {(0, 0): 0}, 0)]
    
    for i in range(1, n):
        new_beam = []
        seen_chains = set()
        
        for chain, occupied, score in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Count new H-H contacts
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                
                # Free space heuristic for future growth
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                # Score: contacts dominate, then free space
                new_score = score + gain * 10.0 + free * 0.1
                
                # Create new chain state
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                # Use a hash of the chain for deduplication
                chain_key = tuple(chain)
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)
                
                new_beam.append((new_chain, new_occupied, new_score))
        
        if not new_beam:
            return None
        
        # Sort by score descending and keep top beam_width
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    
    # Return the best chain from the beam
    if not beam:
        return None
    best = beam[0]
    return best[0]

def _local_search(sequence, chain, rng, max_steps=200):
    """
    Refine the chain using kink moves with simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    is_h = [c == 'H' for c in sequence]
    
    start_temp = 2.0
    end_temp = 0.05
    
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
        
        new_energy = _energy(sequence, chain)
        delta = new_energy - current_energy
        
        if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
            current_energy = new_energy
        else:
            chain[i] = curr
            del occupied[new_pos]
            occupied[curr] = i
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search growth followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(54321 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Determine beam width based on sequence length
    if n <= 30:
        beam_width = 150
    elif n <= 60:
        beam_width = 100
    else:
        beam_width = 80
    
    max_iterations = 2000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Beam search growth
        chain = _beam_grow(sequence, dimension, rng, beam_width)
        if chain is None:
            continue
        
        # Local search refinement
        chain, energy = _local_search(sequence, chain, rng, max_steps=150)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
