import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _init_walk_greedy(sequence, rng):
    n = len(sequence)
    if n == 0:
        return [], {}
    chain = [(0, 0, 0)]
    pos_to_idx = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbours(last):
            if nb not in pos_to_idx:
                free = sum(1 for q in _neighbours(nb) if q not in pos_to_idx)
                options.append((free, nb))
        
        if not options:
            return None, None
        
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.2:
            chosen = options[rng.randint(1, len(options)-1)][1]
        else:
            chosen = options[0][1]
            
        chain.append(chosen)
        pos_to_idx[chosen] = i
        
    return chain, pos_to_idx

def _compute_contacts(sequence, chain, pos_to_idx):
    contacts = set()
    h_positions = {i for i, c in enumerate(sequence) if c == 'H'}
    
    for i in h_positions:
        if i >= len(chain): break
        p = chain[i]
        for nb in _neighbours(p):
            j = pos_to_idx.get(nb)
            if j is not None and j > i + 1 and j in h_positions:
                contacts.add((i, j))
                
    return contacts

def _rotate_point(center, point, axis, angle):
    cx, cy, cz = center
    px, py, pz = point
    rx, ry, rz = px - cx, py - cy, pz - cz
    
    if axis == 0:
        if angle == 90: nrx, nry, nrz = rx, -rz, ry
        elif angle == 180: nrx, nry, nrz = rx, -ry, -rz
        elif angle == 270: nrx, nry, nrz = rx, rz, -ry
    elif axis == 1:
        if angle == 90: nrx, nry, nrz = rz, ry, -rx
        elif angle == 180: nrx, nry, nrz = -rx, ry, -rz
        elif angle == 270: nrx, nry, nrz = -rz, ry, rx
    else:
        if angle == 90: nrx, nry, nrz = -ry, rx, rz
        elif angle == 180: nrx, nry, nrz = -rx, -ry, rz
        elif angle == 270: nrx, nry, nrz = ry, -rx, rz
        
    return (cx + nrx, cy + nry, cz + nrz)

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0, 0, 0]]
        
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    best_contacts = set()
    
    # Initialization
    for _ in range(5):
        init_res = _init_walk_greedy(sequence, rng)
        if init_res[0] is None:
            continue
        chain, pos_to_idx = init_res
        contacts = _compute_contacts(sequence, chain, pos_to_idx)
        energy = -len(contacts)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            best_contacts = contacts
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0
        best_contacts = set()
        
    current_chain = list(best_chain)
    current_energy = best_energy
    current_contacts = set(best_contacts)
    current_pos_to_idx = {p: i for i, p in enumerate(current_chain)}
    
    # SA Parameters
    T_start = 4.0
    T_end = 0.05
    max_steps = 200000  # High cap, time will limit it
    
    step = 0
    while time.time() < _DEADLINE - 1.0:
        frac = step / max_steps
        if frac >= 1.0:
            T = T_end
        else:
            T = T_start * (T_end / T_start) ** frac
            
        # Propose Pivot Move
        if n > 3:
            pivot_idx = rng.randint(1, n - 2)
            axis = rng.randint(0, 2)
            angle = rng.choice([90, 180, 270])
            
            center = current_chain[pivot_idx]
            tail_start = pivot_idx + 1
            tail = current_chain[tail_start:]
            
            new_tail = []
            valid = True
            prefix_positions = set(current_chain[:pivot_idx+1])
            new_positions = set()
            
            for pt in tail:
                new_pt = _rotate_point(center, pt, axis, angle)
                if new_pt in prefix_positions or new_pt in new_positions:
                    valid = False
                    break
                new_positions.add(new_pt)
                new_tail.append(new_pt)
                
            if not valid:
                step += 1
                continue
                
            # Energy Delta
            old_tail_contacts = set()
            for c in current_contacts:
                i, j = c
                if i >= tail_start or j >= tail_start:
                    old_tail_contacts.add(c)
                    
            new_tail_contacts = set()
            new_tail_pos_to_idx = {}
            for k, pt in enumerate(new_tail):
                idx = tail_start + k
                new_tail_pos_to_idx[pt] = idx
                
            for k, new_pt in enumerate(new_tail):
                i = tail_start + k
                if sequence[i] != 'H':
                    continue
                for nb in _neighbours(new_pt):
                    j = current_pos_to_idx.get(nb)
                    if j is not None and j < tail_start:
                        if sequence[j] == 'H' and abs(i - j) > 1:
                            new_tail_contacts.add((min(i, j), max(i, j)))
                    elif nb in new_tail_pos_to_idx:
                        j = new_tail_pos_to_idx[nb]
                        if sequence[j] == 'H' and abs(i - j) > 1:
                            new_tail_contacts.add((min(i, j), max(i, j)))
                            
            delta_E = - (len(new_tail_contacts) - len(old_tail_contacts))
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                # Accept
                current_chain[tail_start:] = new_tail
                
                # Update pos_to_idx
                for pt in tail:
                    del current_pos_to_idx[pt]
                for k, pt in enumerate(new_tail):
                    idx = tail_start + k
                    current_pos_to_idx[pt] = idx
                    
                # Update contacts
                for c in old_tail_contacts:
                    current_contacts.discard(c)
                current_contacts.update(new_tail_contacts)
                
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    best_contacts = set(current_contacts)
                    
        step += 1
        
    return [list(p) for p in best_chain]
