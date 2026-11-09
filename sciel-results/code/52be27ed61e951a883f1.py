import random
import time
import math

_START_TIME = time.time()
DEADLINE = 190.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    n = len(chain)
    if n < 2:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                neighbor = (x+dx, y+dy)
                j = pos_to_idx.get(neighbor)
                if j is not None:
                    if abs(i - j) > 1:
                        if i < j and sequence[j] == 'H':
                            contacts += 1
    return -contacts

def _grow_sequence(sequence, rng):
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    pos_to_idx = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb not in occupied:
                score = rng.random()
                if sequence[i] == 'H':
                    for hnb in _neighbours(nb):
                        if hnb in occupied:
                            idx = pos_to_idx[hnb]
                            if sequence[idx] == 'H':
                                score += 1.0
                options.append((score, nb))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        next_pos = options[0][1]
        chain.append(next_pos)
        occupied.add(next_pos)
        pos_to_idx[next_pos] = i
        
    return chain

def _pivot_move(chain, rng):
    n = len(chain)
    if n < 3:
        return None
    
    pivot_idx = rng.randint(0, n - 2)
    max_len = n - pivot_idx - 1
    if max_len < 1:
        return None
    length = rng.randint(1, max_len)
    
    pivot_point = chain[pivot_idx]
    segment = chain[pivot_idx+1 : pivot_idx+1+length]
    
    rotations = [
        lambda x, y: (pivot_point[0] + (y - pivot_point[1]), pivot_point[1] - (x - pivot_point[0])),
        lambda x, y: (pivot_point[0] - (x - pivot_point[0]), pivot_point[1] - (y - pivot_point[1])),
        lambda x, y: (pivot_point[0] - (y - pivot_point[1]), pivot_point[1] + (x - pivot_point[0]))
    ]
    
    for rot_func in rotations:
        new_segment = [rot_func(x, y) for x, y in segment]
        
        if len(new_segment) != len(set(new_segment)):
            continue
            
        prefix = chain[:pivot_idx+1]
        suffix = chain[pivot_idx+1+length:]
        fixed_part = prefix + suffix
        fixed_set = set(fixed_part)
        
        valid = True
        for p in new_segment:
            if p in fixed_set:
                valid = False
                break
        
        if not valid:
            continue
            
        candidate_chain = prefix + new_segment + suffix
        return candidate_chain
        
    return None

def _slide_move(chain, rng):
    n = len(chain)
    if n < 2:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    
    options = []
    for nb in _neighbours(last):
        if nb == prev:
            continue
        if nb in chain[:-1]:
            continue
        options.append(nb)
        
    if not options:
        return None
        
    new_last = rng.choice(options)
    new_chain = chain[:-1] + [new_last]
    return new_chain

def _anneal(sequence, chain, rng, max_time=5.0):
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain
    best_energy = current_energy
    
    T = 1.0
    start_time = time.time()
    
    while (time.time() - start_time) < max_time:
        if rng.random() < 0.7:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _slide_move(current_chain, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 0.01 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        T *= 0.995
        
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if dimension != 2:
        return [[i, 0] for i in range(n)]

    best_chain = None
    best_energy = 0
    
    remaining = _remaining_time()
    
    # Run 1
    time1 = min(60.0, remaining * 0.6)
    if time1 > 2.0:
        rng = random.Random(42)
        init = _grow_sequence(sequence, rng)
        if not init:
            init = [(i, 0) for i in range(n)]
        
        chain, energy = _anneal(sequence, init, rng, max_time=time1)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    # Run 2
    remaining = _remaining_time()
    if remaining > 20.0:
        time2 = min(40.0, remaining * 0.5)
        rng = random.Random(123)
        init = _grow_sequence(sequence, rng)
        if not init:
            init = [(i, 0) for i in range(n)]
            
        chain, energy = _anneal(sequence, init, rng, max_time=time2)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    # Run 3
    remaining = _remaining_time()
    if remaining > 15.0:
        time3 = min(30.0, remaining * 0.8)
        rng = random.Random(999)
        init = _grow_sequence(sequence, rng)
        if not init:
            init = [(i, 0) for i in range(n)]
            
        chain, energy = _anneal(sequence, init, rng, max_time=time3)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
