import random
import time

_T0 = time.time()
_DEADLINE = 190.0

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied and sequence[occupied[q]] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Try a few greedy starts
    for _ in range(10):
        initial_chain = _grow(sequence, dimension, rng)
        if initial_chain is not None:
            if best_chain is None:
                best_chain = initial_chain
                best_energy = _energy(sequence, best_chain)
            else:
                e = _energy(sequence, initial_chain)
                if e < best_energy:
                    best_chain = initial_chain
                    best_energy = e
                    break # Found a good one, stop greedy

    if best_chain is None:
        return [list(p) for p in [(i, 0) for i in range(len(sequence))]]

    current_chain = best_chain
    current_energy = best_energy
    
    T = 1.0
    cooling = 0.999
    
    while _remaining() > 5.0:
        N = len(sequence)
        move_type = rng.random()
        accepted = False
        new_energy = current_energy
        
        if move_type < 0.7:
            # Kink move
            if N > 2:
                i = rng.randint(1, N - 2)
                pivot = current_chain[i]
                prev = current_chain[i-1]
                next_pos = current_chain[i+1]
                
                candidates = [q for q in _neighbours(pivot) if q != prev and q != next_pos]
                if candidates:
                    new_pos = rng.choice(candidates)
                    # Check collision
                    is_valid = True
                    for k, p in enumerate(current_chain):
                        if k == i + 1:
                            continue
                        if p == new_pos:
                            is_valid = False
                            break
                    if is_valid:
                        new_chain = current_chain.copy()
                        new_chain[i+1] = new_pos
                        new_energy = _energy(sequence, new_chain)
                        
                        delta = new_energy - current_energy
                        if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                            current_chain = new_chain
                            current_energy = new_energy
                            accepted = True
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = current_chain.copy()
        else:
            # Reversal move
            if N > 3:
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 2, N - 1)
                # Check if C[i] and C[j] are neighbors
                if abs(current_chain[i][0] - current_chain[j][0]) + abs(current_chain[i][1] - current_chain[j][1]) == 1:
                    # Check if C[i+1] and C[j+1] are neighbors
                    if abs(current_chain[i+1][0] - current_chain[j+1][0]) + abs(current_chain[i+1][1] - current_chain[j+1][1]) == 1:
                        new_chain = current_chain.copy()
                        segment = current_chain[i+1:j+1]
                        segment.reverse()
                        new_chain[i+1:j+1] = segment
                        
                        new_energy = _energy(sequence, new_chain)
                        delta = new_energy - current_energy
                        if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                            current_chain = new_chain
                            current_energy = new_energy
                            accepted = True
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = current_chain.copy()
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return [list(p) for p in best_chain]
