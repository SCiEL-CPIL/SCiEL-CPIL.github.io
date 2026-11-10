"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None
_CALL_COUNT = 0

def _neighbours(point):
    """Yield 6 neighbors of a point in 3D lattice."""
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y+1, z+1) # dummy to keep structure if needed, but standard 6 is better
    # Actually, let's stick to the efficient 6-neighbor generator
    # The current implementation in the prompt is:
    # yield (x+1, y, z) ... etc
    # Let's use a fast version.
    pass

# Re-implementing _neighbours correctly and fast
def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """
    Greedy growth that prefers H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        # Score candidates by potential H-H contacts
        best_score = -100
        best_site = None
        # Add some randomness to avoid deterministic traps
        for site in candidates:
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H' and abs(i - j) > 1:
                            score += 1
            # Add small random noise
            score += rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_site = site
        
        if best_site is None:
            best_site = rng.choice(candidates)
            
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector v around a given axis (0,1,2) by 90, 180, or 270 degrees.
    """
    x, y, z = v
    if axis == 0: # X axis
        if angle == 1:
            return (x, -z, y)
        elif angle == 2:
            return (x, -y, -z)
        elif angle == 3:
            return (x, z, -y)
    elif axis == 1: # Y axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, z)
        elif angle == 3:
            return (-z, y, x)
    elif axis == 2: # Z axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        elif angle == 3:
            return (-y, x, z)
    return v

def _pivot_move(sequence, chain, i, rng, occupied_set):
    """
    Pivot move: Rotate suffix chain[i:] around chain[i-1].
    Returns (new_chain, energy_delta, new_occupied_set) or None.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    # Choose rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_suffix = []
    for p in suffix:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_suffix.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Calculate energy delta efficiently
    prefix_map = {chain[idx]: idx for idx in range(i)}
    
    old_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    new_occupied = set(new_chain)
    
    return new_chain, delta, new_occupied

def _segment_move(sequence, chain, i, j, rng, occupied_set):
    """
    Segment move: Rotate chain[i+1:j+1] around chain[i].
    Returns (new_chain, energy_delta, new_occupied_set) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    pivot = chain[i]
    segment = chain[i+1:j+1]
    
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_segment = []
    for p in segment:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_segment.append(new_p)
        
    # Check validity
    # The segment must not collide with:
    # 1. Itself (handled by checking new_segment against itself? No, rotation preserves internal distances, but we need to check if the new segment intersects itself? No, rotation is a rigid body motion, so if the original segment was self-avoiding, the new one is too. Wait, the segment is a part of the chain, so it is self-avoiding. The new segment is a rotated copy, so it is also self-avoiding.)
    # 2. The prefix chain[:i+1] (excluding the pivot itself? No, the pivot is the anchor. The segment starts at i+1. The pivot is at i. The first point of the new segment is connected to the pivot. So we must ensure new_segment[0] is not in prefix_set? No, new_segment[0] is adjacent to pivot. It could be in prefix_set? If it is, it's a collision.
    # 3. The suffix chain[j+1:]
    
    # Build set of fixed parts
    fixed_set = set(chain[:i+1]) | set(chain[j+1:])
    # Remove pivot from fixed_set? No, pivot is in chain[:i+1].
    # But new_segment[0] is adjacent to pivot. It's allowed to be adjacent, but not ON top of any fixed point.
    
    # Check new_segment against fixed_set
    for p in new_segment:
        if p in fixed_set:
            return None
            
    # Check new_segment against itself? As argued, rigid rotation of a self-avoiding path is self-avoiding.
    
    new_chain = chain[:i+1] + new_segment + chain[j+1:]
    
    # Calculate energy delta
    # Contacts involving the moved segment change.
    # 1. Old contacts between segment and (prefix + suffix)
    # 2. New contacts between new_segment and (prefix + suffix)
    # 3. Internal contacts of segment are preserved.
    
    # Build map for fixed parts
    fixed_map = {}
    for idx in range(i+1):
        fixed_map[chain[idx]] = idx
    for idx in range(j+1, n):
        fixed_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in fixed_map:
                    j_idx = fixed_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in fixed_map:
                    j_idx = fixed_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    new_occupied = set(new_chain)
    
    return new_chain, delta, new_occupied

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot and Segment moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    current_occupied = set(current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    initial_temp = 5.0
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Choose move type
        if rng.random() < 0.7:
            # Pivot move
            i = rng.randint(1, n - 1)
            result = _pivot_move(sequence, current_chain, i, rng, current_occupied)
        else:
            # Segment move
            # Choose segment length
            max_len = min(20, n - 1)
            seg_len = rng.randint(2, max_len)
            i = rng.randint(0, n - seg_len - 1)
            j = i + seg_len
            result = _segment_move(sequence, current_chain, i, j, rng, current_occupied)
            
        if result is None:
            continue
            
        new_chain, delta, new_occupied = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
            current_occupied = new_occupied
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
            
        # Cooling
        temperature *= 0.999
        
        if temperature < 0.01:
            if no_improve_count > max_no_improve:
                temperature = initial_temp * 0.5
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.1
                no_improve_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    global _CALL_COUNT
    _CALL_COUNT += 1
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 320.0  # Safety margin
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Budget per sequence
    budget = min(remaining, 10.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Initial conformation: Greedy Growth
    chain = _greedy_grow(sequence, dimension, rng)
    if chain is None:
        # Fallback: Random walk
        for _ in range(10):
            chain = _fast_grow_fallback(sequence, dimension, rng)
            if chain is not None:
                break
        if chain is None:
            chain = [(i, 0, 0) for i in range(n)]
            
    # Simulated Annealing
    best_chain, best_energy = _simulated_annealing(sequence, chain, rng, deadline)
    
    # Validation
    if not _is_valid_chain(best_chain):
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

def _fast_grow_fallback(sequence, dimension, rng):
    """Fallback random walk growth."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        if not candidates:
            return None
        site = rng.choice(candidates)
        chain.append(site)
        occupied[site] = i
    return chain

def _is_valid_chain(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if dist != 1:
                return False
    return True

# EVOLVE-BLOCK-END
