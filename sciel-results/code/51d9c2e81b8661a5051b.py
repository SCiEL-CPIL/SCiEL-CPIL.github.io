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

def _rotate_point(p, pivot, axis, angle):
    p = list(p)
    pivot = list(pivot)
    p[0] -= pivot[0]
    p[1] -= pivot[1]
    p[2] -= pivot[2]
    
    if axis == 0:
        if angle == 90: p[1], p[2] = -p[2], p[1]
        elif angle == 180: p[1], p[2] = -p[1], -p[2]
        elif angle == 270: p[1], p[2] = p[2], -p[1]
    elif axis == 1:
        if angle == 90: p[0], p[2] = p[2], -p[0]
        elif angle == 180: p[0], p[2] = -p[0], -p[2]
        elif angle == 270: p[0], p[2] = -p[2], p[0]
    elif axis == 2:
        if angle == 90: p[0], p[1] = -p[1], p[0]
        elif angle == 180: p[0], p[1] = -p[0], -p[1]
        elif angle == 270: p[0], p[1] = p[1], -p[0]
        
    p[0] += pivot[0]
    p[1] += pivot[1]
    p[2] += pivot[2]
    return tuple(p)

def _sa_fold(sequence, dimension, rng, time_limit=5.0):
    start_time = time.time()
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    best_chain = chain[:]
    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    
    n = len(chain)
    if n < 3:
        return best_chain

    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute H positions for fast energy delta? 
    # For simplicity, recalc full energy. N=136, O(N^2) is ~20k ops.
    # 1000 steps * 20k = 20M ops. Might be slow in Python.
    # Let's limit steps or optimize.
    
    steps = 0
    while time.time() - start_time < time_limit and T > min_T:
        i = rng.randint(1, n - 2)
        axis = rng.randint(0, 2)
        angle = rng.choice([90, 180, 270])
        
        new_chain = chain[:]
        pivot = chain[i]
        for j in range(i + 1, n):
            new_chain[j] = _rotate_point(chain[j], pivot, axis, angle)
            
        # Check self-avoidance
        # Fixed: 0..i
        # Moved: i+1..n-1
        occupied_fixed = set(chain[:i+1])
        valid = True
        seen = set()
        for j in range(i + 1, n):
            p = new_chain[j]
            if p in occupied_fixed or p in seen:
                valid = False
                break
            seen.add(p)
            
        if not valid:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling
        steps += 1
        
    return best_chain

def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    # Time budget per sequence: ~9s
    # Let's do multiple restarts
    # If we use SA, each restart takes some time.
    # Let's do 10 restarts of SA, each with 1s time limit?
    # Or 1 long SA run?
    # Multiple short restarts often work better.
    
    time_limit_per_restart = 0.5
    num_restarts = 10
    
    for _ in range(num_restarts):
        chain = _sa_fold(sequence, dimension, rng, time_limit_per_restart)
        if chain is None:
            continue
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
