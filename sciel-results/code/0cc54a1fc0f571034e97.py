"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _rot_x(p, center, sign):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    if sign == 1:
        ndy = dz
        ndz = -dy
    else:
        ndy = -dz
        ndz = dy
    return (center[0] + dx, center[1] + ndy, center[2] + ndz)

def _rot_y(p, center, sign):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    if sign == 1:
        ndx = -dz
        ndz = dx
    else:
        ndx = dz
        ndz = -dx
    return (center[0] + ndx, center[1] + dy, center[2] + ndz)

def _rot_z(p, center, sign):
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    dz = p[2] - center[2]
    if sign == 1:
        ndx = -dy
        ndy = dx
    else:
        ndx = dy
        ndy = -dx
    return (center[0] + ndx, center[1] + ndy, center[2] + dz)

def _get_rotation(p1, p2):
    for axis in range(3):
        for sign in (-1, 1):
            if axis == 0:
                rot = _rot_x(p1, (0,0,0), sign)
            elif axis == 1:
                rot = _rot_y(p1, (0,0,0), sign)
            else:
                rot = _rot_z(p1, (0,0,0), sign)
            if rot == p2:
                return (axis, sign)
    return None

def _initial_grow(sequence, dimension, rng):
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            options.append((free + rng.random(), site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        chosen_site = options[rng.randint(0, k-1)][1]
        chain.append(chosen_site)
        occupied[chosen_site] = i
    return chain

def _calc_energy(sequence, chain):
    coord_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = coord_to_idx.get(q)
                if j is not None and j > i + 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    start_time = time.time()
    deadline = start_time + 235.0
    
    is_h = [c == 'H' for c in sequence]
    
    def get_energy(chain):
        return _calc_energy(sequence, chain)
        
    best_chain = None
    best_energy = 0
    
    # Initial straight line
    straight_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
    best_chain = straight_chain
    best_energy = get_energy(straight_chain)
    
    rng = random.Random(12345 + len(sequence))
    
    while time.time() < deadline:
        init_chain = _initial_grow(sequence, dimension, rng)
        if init_chain is None:
            continue
            
        current_chain = init_chain
        current_energy = get_energy(current_chain)
        
        if current_energy < best_energy:
            best_chain = current_chain
            best_energy = current_energy
            
        # SA Parameters
        t_start = 5.0
        t_end = 0.1
        steps = 2000
        alpha = (t_end / t_start) ** (1.0 / steps)
        
        occupied = {p: i for i, p in enumerate(current_chain)}
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            t = t_start * (alpha ** step)
            
            # Move
            move_type = rng.random()
            new_chain = None
            
            if move_type < 0.8:
                # Pivot
                if n > 2:
                    i = rng.randint(1, n-1)
                    a = current_chain[i-1]
                    b = current_chain[i]
                    # Candidates for b'
                    candidates = []
                    for nb in _neighbours(a):
                        if nb == b:
                            continue
                        if i > 1 and nb == current_chain[i-2]:
                            continue
                        candidates.append(nb)
                        
                    if candidates:
                        b_new = rng.choice(candidates)
                        # Get rotation
                        # Vector a->b
                        v1 = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
                        v2 = (b_new[0]-a[0], b_new[1]-a[1], b_new[2]-a[2])
                        rot = _get_rotation(v1, v2)
                        if rot:
                            axis, sign = rot
                            # Apply rotation to segment [i, n)
                            new_segment = []
                            valid = True
                            # We need to check collisions with [0, i-1)
                            # The fixed part is [0, i-1]
                            fixed_occupied = set(current_chain[:i])
                            
                            for k in range(i, n):
                                p = current_chain[k]
                                if axis == 0:
                                    np = _rot_x(p, a, sign)
                                elif axis == 1:
                                    np = _rot_y(p, a, sign)
                                else:
                                    np = _rot_z(p, a, sign)
                                
                                # Check collision
                                # If k == i, np should be b_new
                                if k > i and np in fixed_occupied:
                                    valid = False
                                    break
                                # Also check against already placed new segment parts
                                # We are building new_segment, so check against it
                                if np in new_segment:
                                    valid = False
                                    break
                                    
                                new_segment.append(np)
                            
                            if valid:
                                # Check if b_new is in fixed_occupied? 
                                # We excluded i-2 and b, but what if b_new is in [0, i-2]?
                                if b_new in fixed_occupied:
                                    valid = False
                                    
                            if valid:
                                new_chain = current_chain[:i] + new_segment
                                # Update occupied? We can recompute energy directly.
                                
            else:
                # Slithering Snake
                # Move tail
                last_idx = n - 1
                prev = current_chain[last_idx - 1]
                last = current_chain[last_idx]
                candidates = []
                for nb in _neighbours(prev):
                    if nb == last:
                        continue
                    # Check if nb is in [0, n-2]
                    if nb not in current_chain[:-1]:
                        candidates.append(nb)
                        
                if candidates:
                    new_last = rng.choice(candidates)
                    new_chain = current_chain[:-1] + [new_last]
                    
            if new_chain is None:
                continue
                
            new_energy = get_energy(new_chain)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / t):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_chain = current_chain
                    best_energy = current_energy
                    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
