"""HP Lattice Protein Folding with Randomized Greedy Growth and Local Search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s limit, 15s safety margin

def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _energy(sequence, chain):
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            r = rng.random()
            score = gain * 10.0 + free * 0.5 + r
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_search(sequence, chain, rng, time_limit):
    if not chain or len(chain) < 4:
        return chain
    
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    start_time = time.time()
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        moves = _neighbours(old_pos)
        rng.shuffle(moves)
        
        for new_pos in moves:
            if new_pos == old_pos:
                continue
                
            j = pos_to_idx.get(new_pos)
            if j is not None:
                if abs(j - i) > 1:
                    continue
            
            old_c = 0
            if sequence[i] == 'H':
                for q in _neighbours(old_pos):
                    k = pos_to_idx.get(q)
                    if k is not None and abs(k - i) > 1 and sequence[k] == 'H':
                        old_c += 1
            
            new_c = 0
            if sequence[i] == 'H':
                for q in _neighbours(new_pos):
                    k = pos_to_idx.get(q)
                    if k is not None and abs(k - i) > 1 and sequence[k] == 'H':
                        new_c += 1
            
            delta = old_c - new_c
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain[i] = new_pos
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                
                break
        
        T *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + n)
    
    elapsed = time.time() - _T0
    remaining = _DEADLINE - elapsed
    if remaining <= 0:
        return [[i, 0] for i in range(n)]
    
    # Budget per call: min(remaining, 15s)
    # But if we have many sequences, we should spend less.
    # Let's assume 10 sequences. 195/10 = ~20s.
    # Use 15s to be safe.
    time_budget = min(remaining, 15.0)
    start_time = time.time()
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Growth
    # Run many fast growths
    num_growths = 1000
    candidates = []
    
    for _ in range(num_growths):
        if time.time() - start_time > time_budget * 0.4:
            break
        chain = _grow(sequence, dimension, rng)
        if chain:
            e = _energy(sequence, chain)
            candidates.append((e, chain))
    
    if not candidates:
        return [[i, 0] for i in range(n)]
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Polish
    # Take top 10 candidates
    K = 10
    for e, chain in candidates[:K]:
        time_left = time_budget - (time.time() - start_time)
        if time_left < 0.1:
            break
        
        # Allocate 80% of remaining time to this polish
        polish_time = time_left * 0.8
        polished_chain = _local_search(sequence, chain, rng, polish_time)
        polished_energy = _energy(sequence, polished_chain)
        
        if best_chain is None or polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy
    
    if best_chain is None:
        best_chain = candidates[0][1]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
