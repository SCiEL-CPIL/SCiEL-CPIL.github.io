import random
import time
import math

def _get_neighbors(pos):
    x, y, z = pos
    return [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]

def _rotate_point(p, center, axis, angle):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    
    if axis == 0:
        if angle == 1:
            new_dy, new_dz = -dz, dy
        elif angle == -1:
            new_dy, new_dz = dz, -dy
        else:
            new_dy, new_dz = -dy, -dz
        return (center[0], center[1] + new_dy, center[2] + new_dz)
    elif axis == 1:
        if angle == 1:
            new_dx, new_dz = dz, -dx
        elif angle == -1:
            new_dx, new_dz = -dz, dx
        else:
            new_dx, new_dz = -dx, -dz
        return (center[0] + new_dx, center[1], center[2] + new_dz)
    else:
        if angle == 1:
            new_dx, new_dy = -dy, dx
        elif angle == -1:
            new_dx, new_dy = dy, -dx
        else:
            new_dx, new_dy = -dx, -dy
        return (center[0] + new_dx, center[1] + new_dy, center[2])

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _get_neighbors(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 290
    
    n = len(sequence)
    if n <= 1:
        return [[0]*dimension for _ in range(n)]
    
    def _random_saw():
        chain = [(0,0,0)]
        occupied = {(0,0,0): 0}
        for i in range(1, n):
            last = chain[-1]
            neighbors = _get_neighbors(last)
            random.shuffle(neighbors)
            placed = False
            for nb in neighbors:
                if nb not in occupied:
                    chain.append(nb)
                    occupied[nb] = i
                    placed = True
                    break
            if not placed:
                return None
        return chain

    best_chain = None
    best_energy = 0
    
    for _ in range(10):
        chain = _random_saw()
        if chain is None: continue
        e = _energy(sequence, chain)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e
            
    if best_chain is None:
        best_chain = [(i,0,0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        
    chain = best_chain[:]
    current_energy = best_energy
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        k = random.randint(1, n-2)
        axis = random.randint(0, 2)
        angle = random.choice([1, -1, 2])
        
        center = chain[k]
        
        old_tail = chain[k+1:]
        new_tail = [_rotate_point(p, center, axis, angle) for p in old_tail]
        
        new_pos_set = set(new_tail)
        if len(new_pos_set) != len(new_tail):
            continue
            
        valid = True
        for new_p in new_tail:
            if new_p in pos_to_idx:
                idx = pos_to_idx[new_p]
                if idx <= k:
                    valid = False
                    break
        if not valid:
            continue
            
        delta_e = 0
        for i, new_p in enumerate(new_tail):
            res_idx = k + 1 + i
            if sequence[res_idx] == 'H':
                for nb in _get_neighbors(new_p):
                    if nb in pos_to_idx:
                        nb_idx = pos_to_idx[nb]
                        if nb_idx <= k and sequence[nb_idx] == 'H':
                            delta_e += 1
                            
        for i, old_p in enumerate(old_tail):
            res_idx = k + 1 + i
            if sequence[res_idx] == 'H':
                for nb in _get_neighbors(old_p):
                    if nb in pos_to_idx:
                        nb_idx = pos_to_idx[nb]
                        if nb_idx <= k and sequence[nb_idx] == 'H':
                            delta_e -= 1
                            
        new_energy = current_energy - delta_e
        
        if new_energy < current_energy:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp((current_energy - new_energy) / T)
            else:
                accept = False
                
        if accept:
            chain[k+1:] = new_tail
            for p in old_tail:
                del pos_to_idx[p]
            for i, p in enumerate(new_tail):
                pos_to_idx[p] = k + 1 + i
                
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return [list(p) for p in best_chain]
