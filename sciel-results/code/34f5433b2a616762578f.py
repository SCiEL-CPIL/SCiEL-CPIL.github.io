import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _initial_fold(sequence, dimension, rng):
    """Greedy growth to create a valid initial self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Bias towards H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        gain += 1
            # Bias towards free space to avoid trapping
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 5.0 + free * 0.1 + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
            
        best_site = max(candidates, key=lambda x: x[0])[1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain, occupied

def _energy(sequence, chain, occupied):
    """Calculate energy (negative number of H-H contacts)."""
    contacts = 0
    h_positions = [i for i, res in enumerate(sequence) if res == 'H']
    
    for i in h_positions:
        p = chain[i]
        for q in _neighbours(p):
            j = occupied.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _try_regrow(chain, occupied, sequence, dimension, rng, cut):
    """
    Cut the chain at index 'cut' and regrow the tail.
    Returns (new_chain, new_occupied, success)
    """
    n = len(chain)
    
    # Remove old tail from occupied
    for i in range(cut, n):
        del occupied[chain[i]]
        
    new_chain = chain[:cut]
    current_pos = chain[cut-1]
    
    # We need to ensure we don't collide with the new tail parts.
    # Since we add to `occupied` immediately, `site in occupied` check works.
    
    for i in range(cut, n):
        candidates = []
        for site in _neighbours(current_pos):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 5.0 + free * 0.1 + rng.random()
            candidates.append((score, site))
            
        if not candidates:
            # Failed to regrow, restore
            for i in range(cut, n):
                occupied[chain[i]] = i
            return None, None, False
            
        best_site = max(candidates, key=lambda x: x[0])[1]
        new_chain.append(best_site)
        occupied[best_site] = i
        current_pos = best_site
        
    return new_chain, occupied, True

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
        
    rng = random.Random(12345 + len(sequence))
    
    # Time budget per sequence
    # Total 330s for 26 sequences.
    # Let's use 11s per sequence to be safe (26 * 11 = 286s).
    time_limit = 11.0
    start_time = time.time()
    
    # 1. Initial Greedy Fold
    init_result = _initial_fold(sequence, dimension, rng)
    if init_result is None:
        # Fallback: straight line
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}
    else:
        chain, occupied = init_result
        
    current_energy = _energy(sequence, chain, occupied)
    best_chain = chain[:]
    best_energy = current_energy
    
    # 2. Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Choose move type
        # 60% Regrow, 40% Slide
        if rng.random() < 0.6:
            # Regrow
            if n < 5:
                cut = 1
            else:
                min_cut = max(2, int(n * 0.2))
                max_cut = min(n - 1, int(n * 0.8))
                if min_cut >= max_cut:
                    cut = n // 2
                else:
                    cut = rng.randint(min_cut, max_cut)
                    
            new_chain, new_occupied, success = _try_regrow(chain, occupied, sequence, dimension, rng, cut)
            if not success:
                continue
                
            new_energy = _energy(sequence, new_chain, new_occupied)
            delta_e = new_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                occupied = new_occupied
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
        else:
            # Slide
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            
            p = chain[i]
            prev = chain[i-1]
            next_ = chain[i+1]
            
            candidates = []
            for site in _neighbours(p):
                if site == prev or site == next_:
                    continue
                if site in occupied:
                    continue
                
                delta = 0
                if sequence[i] == 'H':
                    # Remove old
                    for q in _neighbours(p):
                        j = occupied.get(q)
                        if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                            delta -= 1
                    # Add new
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                            delta += 1
                            
                candidates.append((delta, site))
                
            if not candidates:
                continue
                
            # Pick random candidate
            delta_e, new_site = rng.choice(candidates)
            
            # Perform move
            del occupied[p]
            occupied[new_site] = i
            new_chain = chain[:]
            new_chain[i] = new_site
            
            # Metropolis criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            
            # Cooling
            T *= alpha
            if T < T_min:
                T = T_min
                
    return [list(p) for p in best_chain]
