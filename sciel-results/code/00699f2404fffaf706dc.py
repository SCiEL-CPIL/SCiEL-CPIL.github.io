import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, res in enumerate(sequence):
        if res == "H":
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, time_limit):
    N = len(sequence)
    best_energy = _calc_energy(sequence, chain)
    best_chain = chain[:]
    
    current_chain = chain[:]
    current_energy = best_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    start_time = time.time()
    rng = random.Random(hash(sequence))
    
    while time.time() - start_time < time_limit:
        if N < 3:
            break
            
        p = rng.randint(0, N-2)
        e = rng.randint(p+1, N-1)
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        
        # Pivot move
        pivot = current_chain[p]
        old_segment_positions = set(current_chain[p+1:e+1])
        
        new_positions = []
        valid = True
        for i in range(p+1, e+1):
            v = [current_chain[i][k] - pivot[k] for k in range(dimension)]
            
            other_axes = [a for a in range(dimension) if a != axis]
            a1, a2 = other_axes[0], other_axes[1]
            
            if direction == 1:
                v[a1], v[a2] = v[a2], -v[a1]
            else:
                v[a1], v[a2] = -v[a2], v[a1]
                
            new_p = tuple(pivot[k] + v[k] for k in range(dimension))
            new_positions.append(new_p)
            
            if new_p in occupied and new_p not in old_segment_positions:
                valid = False
                break
                
        if valid:
            new_chain = current_chain[:p+1] + new_positions + current_chain[e+1:]
            new_energy = _calc_energy(sequence, new_chain)
            
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                occupied = {p: i for i, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
                    
    return best_chain, best_energy

def fold(sequence, dimension):
    N = len(sequence)
    time_per_sequence = 12.0
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    start_time = time.time()
    
    while time.time() - start_time < time_per_sequence:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        elapsed = time.time() - start_time
        remaining = time_per_sequence - elapsed
        if remaining < 0.1:
            break
            
        search_time = min(2.0, remaining)
        
        improved_chain, improved_energy = _local_search(sequence, chain, search_time)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(N)]
        
    return [list(p) for p in best_chain]
