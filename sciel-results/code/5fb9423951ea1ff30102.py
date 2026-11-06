import random
import time
import math

# Precompute 90-degree rotation matrices for 3D
_ROT_MATRICES = [
    # axis=0 (x), dir=+1: (x, y, z) -> (x, -z, y)
    [[1,0,0],[0,0,-1],[0,1,0]],
    # axis=0, dir=-1: (x, y, z) -> (x, z, -y)
    [[1,0,0],[0,0,1],[0,-1,0]],
    # axis=1 (y), dir=+1: (x, y, z) -> (z, y, -x)
    [[0,0,1],[0,1,0],[-1,0,0]],
    # axis=1, dir=-1: (x, y, z) -> (-z, y, x)
    [[0,0,-1],[0,1,0],[1,0,0]],
    # axis=2 (z), dir=+1: (x, y, z) -> (-y, x, z)
    [[0,-1,0],[1,0,0],[0,0,1]],
    # axis=2, dir=-1: (x, y, z) -> (y, -x, z)
    [[0,1,0],[-1,0,0],[0,0,1]],
]

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _grow(sequence, dimension, rng):
    """Greedy growth to get a valid initial chain."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i-1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random() * 0.1
            options.append((score, site))
        if not options:
            return None
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _energy(sequence, chain):
    """Compute the energy (negative contact count) of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _rotate_segment(segment, pivot, axis, direction, dimension):
    """Rotate a segment around a pivot point by 90 degrees."""
    mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
    new_segment = []
    for p in segment:
        v = [p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]]
        # Matrix multiplication
        nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
        nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
        nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
        new_segment.append((pivot[0]+nv0, pivot[1]+nv1, pivot[2]+nv2))
    return new_segment

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # Global time management. The contract says 330s wall-clock for the whole evaluation.
    # We assume `fold` is called once per sequence, but to be safe, we track a global budget.
    # However, since we don't know the call order or if it's the same process, 
    # and the contract says "fold is called once per benchmark sequence in a sandboxed subprocess",
    # we can assume a fresh process for each call? 
    # "fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together."
    # This implies 240s TOTAL for all sequences if they are in the same subprocess.
    # If they are separate subprocesses, it's ambiguous. 
    # "The evaluator measures the raw value... Each evaluation must finish within 300 s."
    # Usually, this means the whole run is limited. 
    # Let's use a global start time to track remaining budget.
    
    # To be safe against a shared process, we'll use a global deadline.
    # If this is a fresh process, we have ~300s. If shared, we use up the budget.
    # We'll define a global start time at module load.
    
    global _GLOBAL_START_TIME
    if not hasattr(fold, '_GLOBAL_START_TIME'):
        fold._GLOBAL_START_TIME = time.time()
    
    # Budget: 280 seconds to be safe (leaving 20s for overhead/other sequences if shared, or just safety margin).
    # If each call is separate, we have ~300s each. 
    # Let's check the contract again: "240 s CPU budget for all sequences together".
    # This strongly implies a single process or a strict total limit.
    # We will use 260s as the deadline for the entire program execution.
    
    deadline = fold._GLOBAL_START_TIME + 260.0
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Initial growth
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback: random walk? Or just return straight line if we can't grow?
        # A straight line is always valid.
        chain = [(i, 0, 0) for i in range(n)]
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain.copy()
    
    # SA Parameters
    # We want to run as many moves as possible.
    # Pivot move selection is O(1) but rotation/collision check is O(n).
    # Energy delta is O(n) because we need to check boundary contacts.
    
    # Let's optimize the inner loop.
    # We can precompute the H status of the chain.
    # We maintain `occupied` dict.
    
    # Temperature schedule
    # We don't know the total number of steps in advance.
    # We will use a time-based temperature reduction.
    # T = T_max * exp(-k * t_elapsed)
    # Or linear.
    
    T_max = 5.0
    T_min = 0.01
    
    # We will run until deadline.
    while time.time() < deadline:
        # Choose pivot
        k = rng.randint(1, n-1)
        pivot_site = chain[k-1]
        
        # Segment to rotate: chain[k:]
        segment = chain[k:]
        seg_len = len(segment)
        
        # Remove segment from occupied
        for p in segment:
            del occupied[p]
            
        # Choose rotation
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        
        new_segment = _rotate_segment(segment, pivot_site, axis, direction, dimension)
        
        # Check collision
        valid = True
        for p in new_segment:
            if p in occupied:
                valid = False
                break
        
        if not valid:
            # Restore
            for i, p in enumerate(segment):
                occupied[p] = k + i
            continue
            
        # Calculate Energy Delta
        # Only boundary contacts change.
        # Old contacts: between segment and R (chain[0..k-1])
        # New contacts: between new_segment and R
        
        old_contacts = 0
        new_contacts = 0
        
        # Iterate over segment
        for i, p in enumerate(segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in occupied:
                        nb_idx = occupied[nb]
                        if is_h[nb_idx]:
                            old_contacts += 1
        
        # Iterate over new_segment
        for i, p in enumerate(new_segment):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(p):
                    if nb in occupied:
                        nb_idx = occupied[nb]
                        if is_h[nb_idx]:
                            new_contacts += 1
        
        delta_contacts = new_contacts - old_contacts
        delta_energy = -delta_contacts
        
        # Acceptance
        # Calculate T based on time elapsed since start of this fold call
        t_elapsed = time.time() - fold._GLOBAL_START_TIME
        # Assume total budget is 260s.
        # T = T_max * (1 - t_elapsed / 260.0)
        # This is a linear cooling over the total budget.
        # However, if this is the 1st sequence, we have 260s left. If 10th, we have 160s.
        # This is a global cooling schedule. It might be too aggressive for later sequences.
        # Better: Cool based on local progress? But we don't know how many sequences remain.
        # Let's use a local cooling schedule for each call, assuming each call gets ~10s?
        # No, the budget is shared.
        
        # Let's just use a constant T for a while, then reduce?
        # Or use a global T that depends on total time elapsed.
        
        # If we use global time, T drops to 0 quickly for later sequences if they start late?
        # No, T_max is 5. At t=0, T=5. At t=260, T=0.
        # If sequence 10 starts at t=100, T starts at 5 * (1 - 100/260) ~ 3.
        # This seems reasonable.
        
        T = T_max * (1.0 - t_elapsed / 260.0)
        if T < T_min:
            T = T_min
            
        if delta_energy < 0:
            accept = True
        else:
            prob = math.exp(-delta_energy / T)
            accept = rng.random() < prob
            
        if accept:
            # Update
            for i, p in enumerate(new_segment):
                occupied[p] = k + i
                chain[k + i] = p
            current_energy += delta_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain.copy()
        else:
            # Restore
            for i, p in enumerate(segment):
                occupied[p] = k + i
            # chain is already correct
            
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
