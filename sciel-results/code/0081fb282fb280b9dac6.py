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
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _fast_grow(sequence, dimension, rng):
    """
    Fast randomized growth.
    Tries to grow the chain by picking a random valid neighbor.
    If stuck, backtracks (simple random walk with backtracking).
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
        
        site = rng.choice(candidates)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _initial_conformation(sequence, dimension, rng):
    """
    Generate an initial valid conformation.
    Uses a spiral-like or compact random walk.
    """
    n = len(sequence)
    for _ in range(10):
        chain = _fast_grow(sequence, dimension, rng)
        if chain is not None:
            return chain
    
    # Fallback: Straight line (valid but bad energy)
    return [(i, 0, 0) for i in range(n)]

def _rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector v around a given axis (0,1,2) by 90, 180, or 270 degrees.
    Returns the new vector.
    """
    x, y, z = v
    if axis == 0: # X axis
        if angle == 1: # 90 deg
            return (x, -z, y)
        elif angle == 2: # 180 deg
            return (x, -y, -z)
        elif angle == 3: # 270 deg
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

def _pivot_move(sequence, chain, i, rng):
    """
    Pivot move: Rotate suffix chain[i:] around chain[i-1].
    Returns (new_chain, energy_delta) or None.
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
    
    # Calculate energy delta
    prefix_map = {}
    for idx in range(i):
        prefix_map[chain[idx]] = idx
    
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
    
    return new_chain, delta

def _is_valid_chain(chain):
    """
    Check if the chain is self-avoiding and has unit steps.
    """
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

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot moves only (Segment/Reversal were invalid).
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_end = deadline
    initial_temp = 5.0
    
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < t_end:
        # Only use Pivot moves as they preserve connectivity
        i = rng.randint(1, n - 1)
        result = _pivot_move(sequence, current_chain, i, rng)
        
        if result is None:
            # Failed move (collision), continue
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
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
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        # Not enough time, return a simple valid conformation
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Heuristic: Assume we have ~1/26th of the total time left on average?
    # Or just use a fixed budget per sequence if possible.
    # Total budget 330s. 26 sequences. ~12s per sequence.
    # Let's use 10s per sequence to be safe.
    budget = min(remaining, 10.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    # Initial conformation
    chain = _initial_conformation(sequence, dimension, rng)
    if chain is None:
        chain = [(i, 0, 0) for i in range(n)]
        
    # Simulated Annealing
    best_chain, best_energy = _simulated_annealing(sequence, chain, rng, deadline)
    
    # Validation: Ensure the chain is valid (connected and self-avoiding)
    if not _is_valid_chain(best_chain):
        # Fallback to a valid straight line if SA produced an invalid chain (should not happen with Pivot, but safe)
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Convert to lists of ints
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
