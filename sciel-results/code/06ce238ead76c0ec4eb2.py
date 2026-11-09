"""HP lattice protein folding using Simulated Annealing with local moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None
DEADLINE = 190.0

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _remaining_time():
    return DEADLINE - (time.time() - _get_start_time())

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow_greedy(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start}
    pos_to_idx = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = pos_to_idx[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        # Sort by score descending and pick the best, or pick randomly from top 3?
        # Greedy: pick best.
        options.sort(reverse=True)
        site = options[0][1]
        
        occupied.add(site)
        pos_to_idx[site] = i
        chain.append(site)
        
    return chain

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    if _remaining_time() < 5.0:
        chain = _grow_greedy(sequence, dimension, random.Random(12345 + len(sequence)))
        if chain is None:
            chain = [(i, 0) for i in range(len(sequence))]
        return [list(p) for p in chain]

    rng = random.Random(12345 + len(sequence) + int(time.time() % 1000))
    
    # 1. Generate initial chain
    best_chain, best_energy = None, 1
    for _ in range(20):
        chain = _grow_greedy(sequence, dimension, rng)
        if chain is None:
            continue
        e = _compute_energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(len(sequence))]
        best_energy = _compute_energy(sequence, best_chain)

    # 2. Simulated Annealing
    current_chain = list(best_chain)
    current_energy = best_energy
    
    n = len(sequence)
    
    T = 1.0
    cooling_rate = 0.9995
    
    while _remaining_time() > 1.0 and T > 0.01:
        move_type = rng.choice(['pivot', 'reversal', 'slither'])
        
        new_chain = list(current_chain)
        
        if move_type == 'pivot':
            k = rng.randint(1, n - 1)
            origin = new_chain[k-1]
            segment = new_chain[k:]
            
            # Rotate segment
            rot = rng.choice([0, 1, 2, 3])
            new_segment = []
            for (x, y) in segment:
                dx, dy = x - origin[0], y - origin[1]
                if rot == 0:
                    new_segment.append((origin[0] + dx, origin[1] + dy))
                elif rot == 1:
                    new_segment.append((origin[0] - dy, origin[1] + dx))
                elif rot == 2:
                    new_segment.append((origin[0] - dx, origin[1] - dy))
                else:
                    new_segment.append((origin[0] + dy, origin[1] - dx))
            
            new_chain[k:] = new_segment
            
        elif move_type == 'reversal':
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i > j: i, j = j, i
            new_chain[i:j+1] = new_chain[i:j+1][::-1]
            
        else: # slither
            last = new_chain[-1]
            moves = list(_neighbours(last))
            rng.shuffle(moves)
            valid_move = None
            for m in moves:
                if m not in set(new_chain[:-1]):
                    valid_move = m
                    break
            if valid_move is None:
                continue
            new_chain[-1] = valid_move

        # Check validity of new_chain
        # Self-avoidance
        if len(set(new_chain)) != n:
            continue
            
        # Calculate energy
        new_energy = _compute_energy(sequence, new_chain)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_rate

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
