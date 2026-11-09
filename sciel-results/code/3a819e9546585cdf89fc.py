import random
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 190.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

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

def _init_chain(sequence, rng):
    L = len(sequence)
    if L == 0: return []
    if L == 1: return [(0,0)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for i in range(1, L):
        res = sequence[i]
        options = []
        for nb in _neighbours(chain[-1]):
            if nb in occupied:
                continue
            score = rng.random()
            if res == 'H':
                for nnb in _neighbours(nb):
                    if nnb in occupied:
                        score += 1.0
            options.append((score, nb))
            
        if not options:
            return None
        
        options.sort(reverse=True)
        top_k = min(3, len(options))
        chosen = rng.choice(options[:top_k])
        next_point = chosen[1]
        chain.append(next_point)
        occupied.add(next_point)
        
    return chain

def fold(sequence, dimension):
    L = len(sequence)
    if L == 0:
        return []
    
    rng = random.Random(42 + L)
    
    best_chain = None
    best_energy = 0
    attempts = 0
    while best_chain is None and attempts < 100:
        best_chain = _init_chain(sequence, rng)
        attempts += 1
        
    if best_chain is None:
        best_chain = [(i, 0) for i in range(L)]
        
    best_energy = _energy(sequence, best_chain)
    
    current_chain = best_chain
    current_energy = best_energy
    occupied = set(current_chain)
    
    T = 1.0
    cooling_rate = 0.999
    
    while _time_left() > 0.1:
        move_type = rng.random()
        
        if move_type < 0.6:
            # Flip
            if L < 4:
                T *= cooling_rate
                continue
                
            i = rng.randint(0, L - 4)
            j = rng.randint(i + 2, L - 1)
            
            pivot = current_chain[i]
            segment = current_chain[i+1:j+1]
            
            # Try 3 rotations
            moved = False
            for rot in range(1, 4):
                # Rotate segment around pivot
                new_segment = []
                for p in segment:
                    dx = p[0] - pivot[0]
                    dy = p[1] - pivot[1]
                    for _ in range(rot):
                        dx, dy = dy, -dx
                    new_segment.append((pivot[0] + dx, pivot[1] + dy))
                
                # Check collision
                # Remove old segment from occupied
                for p in segment:
                    occupied.remove(p)
                
                ok = True
                for p in new_segment:
                    if p in occupied:
                        ok = False
                        break
                if ok:
                    seen = set()
                    for p in new_segment:
                        if p in seen:
                            ok = False
                            break
                        seen.add(p)
                
                if ok:
                    # Check connections
                    if abs(new_segment[0][0] - pivot[0]) + abs(new_segment[0][1] - pivot[1]) != 1:
                        ok = False
                    else:
                        if j < L - 1:
                            next_p = current_chain[j+1]
                            if abs(new_segment[-1][0] - next_p[0]) + abs(new_segment[-1][1] - next_p[1]) != 1:
                                ok = False
                
                if ok:
                    # Add new segment to occupied
                    for p in new_segment:
                        occupied.add(p)
                    
                    new_chain = current_chain[:i+1] + new_segment + current_chain[j+1:]
                    new_energy = _energy(sequence, new_chain)
                    
                    delta_e = new_energy - current_energy
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        occupied = set(current_chain)
                        
                        if current_energy < best_energy:
                            best_chain = current_chain
                            best_energy = current_energy
                        moved = True
                        break
                    else:
                        # Reject, remove new, restore old
                        for p in new_segment:
                            occupied.remove(p)
                        for p in segment:
                            occupied.add(p)
                else:
                    # Invalid, restore old
                    for p in segment:
                        occupied.add(p)
            
            if not moved:
                # Ensure occupied is correct if we broke out early or didn't move
                # If we tried rotations and none worked, occupied should be restored
                # The loop above restores occupied on failure.
                pass
                
        else:
            # RGD
            if L < 4:
                T *= cooling_rate
                continue
                
            i = rng.randint(0, L - 4)
            j = rng.randint(i + 2, L - 1)
            
            segment = current_chain[i+1:j+1]
            for p in segment:
                occupied.remove(p)
            
            new_segment = []
            current = current_chain[i]
            ok = True
            
            for k in range(i + 1, j + 1):
                res = sequence[k]
                options = []
                for nb in _neighbours(current):
                    if nb in occupied:
                        continue
                    score = rng.random()
                    if res == 'H':
                        for nnb in _neighbours(nb):
                            if nnb in occupied:
                                score += 1.0
                    options.append((score, nb))
                
                if not options:
                    ok = False
                    break
                
                options.sort(reverse=True)
                top_k = min(3, len(options))
                chosen = rng.choice(options[:top_k])
                next_p = chosen[1]
                new_segment.append(next_p)
                occupied.add(next_p)
                current = next_p
            
            if not ok:
                # Restore
                for p in new_segment:
                    occupied.remove(p)
                for p in segment:
                    occupied.add(p)
                T *= cooling_rate
                continue
            
            # Check tail connection
            if j < L - 1:
                next_p = current_chain[j+1]
                if abs(new_segment[-1][0] - next_p[0]) + abs(new_segment[-1][1] - next_p[1]) != 1:
                    for p in new_segment:
                        occupied.remove(p)
                    for p in segment:
                        occupied.add(p)
                    T *= cooling_rate
                    continue
            
            new_chain = current_chain[:i+1] + new_segment + current_chain[j+1:]
            new_energy = _energy(sequence, new_chain)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_chain = new_chain
                current_energy = new_energy
                occupied = set(current_chain)
                
                if current_energy < best_energy:
                    best_chain = current_chain
                    best_energy = current_energy
            else:
                for p in new_segment:
                    occupied.remove(p)
                for p in segment:
                    occupied.add(p)
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return [list(p) for p in best_chain]
