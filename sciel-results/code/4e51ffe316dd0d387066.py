import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _grow(sequence, dimension, rng):
    """
    Greedy growth with some randomization to avoid getting stuck.
    Returns a list of tuples representing the chain.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
        
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score the site
            # 1. Immediate H-H contact gain
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        # Check if it's a chain neighbor (i-1)
                        if idx == i - 1:
                            continue
                        if sequence[idx] == 'H':
                            gain += 1
            
            # 2. Free space (to avoid trapping)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # 3. Random tie-breaker
            rand = rng.random()
            
            # Weighted score: Gain is most important, free space second
            score = gain * 10.0 + free * 0.1 + rand
            options.append((score, site))
            
        if not options:
            return None
            
        # Pick best
        best_site = max(options)[1]
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total energy of a chain."""
    if not chain:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [1 if s == 'H' else 0 for s in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _calc_delta_energy(sequence, chain, k, new_segment, occupied_rest, is_h):
    """
    Calculate the change in energy when replacing segment k: of chain with new_segment.
    occupied_rest contains indices 0..k-1.
    """
    # We will just recalculate the full energy in the main loop for simplicity and speed
    # if N is not too large. This function is kept for interface compatibility if needed,
    # but the main loop will use _energy.
    return 0

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    max_time = 11.5
    start_time = time.time()
    
    rng = random.Random(42 + len(sequence))
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0
    
    # Precompute rotation logic inline for speed
    
    while True:
        if time.time() - start_time > max_time:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
            
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
            
        T = 2.0
        
        while True:
            if time.time() - start_time > max_time * 0.95:
                break
            
            move_type = rng.random()
            k = rng.randint(1, n - 1)
            
            old_segment = chain[k:]
            
            for p in old_segment:
                del occupied[p]
            
            new_segment = None
            valid = True
            
            if move_type < 0.7:
                # PIVOT
                pivot_site = chain[k-1]
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                
                new_segment = []
                for p in old_segment:
                    x, y, z = p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]
                    if axis == 0:
                        if direction == 1:
                            nx, ny, nz = x, -z, y
                        else:
                            nx, ny, nz = x, z, -y
                    elif axis == 1:
                        if direction == 1:
                            nx, ny, nz = z, y, -x
                        else:
                            nx, ny, nz = -z, y, x
                    else:
                        if direction == 1:
                            nx, ny, nz = -y, x, z
                        else:
                            nx, ny, nz = y, -x, z
                    
                    new_p = (pivot_site[0]+nx, pivot_site[1]+ny, pivot_site[2]+nz)
                    new_segment.append(new_p)
                    if new_p in occupied:
                        valid = False
                        break
            else:
                # MOVE
                axis = rng.randint(0, 2)
                direction = rng.choice([-1, 1])
                ox, oy, oz = 0, 0, 0
                if axis == 0: ox = direction
                elif axis == 1: oy = direction
                else: oz = direction
                
                new_segment = []
                for p in old_segment:
                    new_p = (p[0]+ox, p[1]+oy, p[2]+oz)
                    new_segment.append(new_p)
                    if new_p in occupied:
                        valid = False
                        break
            
            if not valid:
                for i, p in enumerate(old_segment):
                    occupied[p] = k+i
                continue
            
            # Apply and recalc energy
            for i, p in enumerate(new_segment):
                chain[k+i] = p
                occupied[p] = k+i
            
            new_energy = _energy(sequence, chain)
            delta_energy = new_energy - current_energy
            
            if delta_energy < 0 or (rng.random() < math.exp(-delta_energy / T)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Revert
                for i, p in enumerate(old_segment):
                    chain[k+i] = p
                    occupied[p] = k+i

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
