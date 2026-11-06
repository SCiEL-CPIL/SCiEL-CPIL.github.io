import random
import time
import math

_T0 = time.time()
_DEADLINE = 170.0  # Safety margin

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

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

def _grow(sequence, dimension, rng):
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + free + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        site = candidates[0][1]
        chain.append(site)
        occupied[site] = i
        
    return chain

def _sa_optimize(sequence, chain, occupied, time_limit, rng):
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = chain[:]
    current_occupied = occupied.copy()
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = current_chain[:]
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # 1-step move
        i = rng.randint(0, n - 1)
        p_old = current_chain[i]
        
        free_neighbors = [q for q in _neighbours(p_old) if q not in current_occupied]
        if not free_neighbors:
            T *= cooling
            continue
            
        p_new = rng.choice(free_neighbors)
        
        # Check validity: p_new cannot be adjacent to chain[i-1] or chain[i+1]
        valid = True
        if i > 0 and p_new == current_chain[i - 1]:
            valid = False
        if i < n - 1 and p_new == current_chain[i + 1]:
            valid = False
            
        if not valid:
            T *= cooling
            continue
        
        # Calculate Delta Energy
        delta = 0
        if sequence[i] == "H":
            # Loss of contacts at p_old
            for q in _neighbours(p_old):
                j = current_occupied.get(q, -1)
                if j != -1 and abs(i - j) > 1 and sequence[j] == "H":
                    delta += 1 # Losing contact -> Energy increases
            
            # Gain of contacts at p_new
            for q in _neighbours(p_new):
                j = current_occupied.get(q, -1)
                if j != -1 and j != i and abs(i - j) > 1 and sequence[j] == "H":
                    delta -= 1 # Gaining contact -> Energy decreases
        
        # Accept/Reject
        if delta <= 0:
            current_occupied.pop(p_old)
            current_occupied[p_new] = i
            current_chain[i] = p_new
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            if rng.random() < math.exp(-delta / T):
                current_occupied.pop(p_old)
                current_occupied[p_new] = i
                current_chain[i] = p_new
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain[:]
        
        T *= cooling
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    elapsed = time.time() - _T0
    remaining_time = _DEADLINE - elapsed
    if remaining_time <= 0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(42)
    
    best_chain = None
    best_energy = 0
    
    init_chain = _grow(sequence, dimension, rng)
    if init_chain is None:
        init_chain = [(i, 0) for i in range(n)]
    
    init_occupied = {p: i for i, p in enumerate(init_chain)}
    init_energy = _compute_energy(sequence, init_chain)
    
    best_chain = init_chain
    best_energy = init_energy
    
    num_restarts = 10
    time_per_restart = remaining_time / num_restarts * 0.8
    
    for r in range(num_restarts):
        if time.time() - _T0 > _DEADLINE - 5:
            break
        
        if r == 0:
            chain = init_chain
            occ = init_occupied
        else:
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                chain = init_chain
                occ = init_occupied
            else:
                occ = {p: i for i, p in enumerate(chain)}
        
        res_chain, res_energy = _sa_optimize(sequence, chain, occ, time_per_restart, rng)
        
        if res_energy < best_energy:
            best_energy = res_energy
            best_chain = res_chain
            
    return [list(p) for p in best_chain]
