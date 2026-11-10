import random
import time
import math

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
        if sequence[i] != 'H':
            continue
        for q in _neighbours(p):
            if q in where:
                j = where[q]
                if abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _rotate_vector(v, axis, angle):
    """Rotate vector v around axis by angle (1=90, 2=180, 3=270 degrees)."""
    x, y, z = v
    if axis == 0:  # x-axis
        if angle == 1:
            return (x, -z, y)
        elif angle == 2:
            return (x, -y, -z)
        else:
            return (x, z, -y)
    elif axis == 1:  # y-axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, -z)
        else:
            return (-z, y, x)
    else:  # z-axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        else:
            return (-y, x, z)

def _pivot_move(chain, i, axis, angle):
    """Rotate chain[i+1:] around chain[i] parallel to axis."""
    n = len(chain)
    if i < 1 or i >= n - 1:
        return None
    
    pivot = chain[i]
    new_chain = list(chain[:i+1])
    
    for p in chain[i+1:]:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        rv = _rotate_vector(v, axis, angle)
        new_chain.append((rv[0] + pivot[0], rv[1] + pivot[1], rv[2] + pivot[2]))
    
    return new_chain

def _is_valid_chain(chain, i):
    """Check if chain is self-avoiding. For pivot move at index i, only need to check tail vs head."""
    n = len(chain)
    if n <= 2:
        return True
    
    # For efficiency in pivot moves, we can check only the changed part
    # But for general use, full check
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    return True

def _fast_grow(sequence, dimension=3):
    """Greedy growth with backtracking to create initial self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin, grow in a roughly linear fashion to avoid immediate collisions
    # Use a simple strategy: go +x, then +y, then +z, etc.
    # Better: use a snake pattern or just random walk with backtracking
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    # Directions in order of preference for initial growth
    directions = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
    
    for step in range(1, n):
        last = chain[-1]
        found = False
        # Try directions, with some randomization to avoid deterministic traps
        dirs = list(directions)
        random.shuffle(dirs)
        
        for d in dirs:
            new_p = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            if new_p not in occupied:
                chain.append(new_p)
                occupied.add(new_p)
                found = True
                break
        
        if not found:
            # Backtrack: this simple approach doesn't support deep backtracking
            # For a better initial structure, we might need a more sophisticated method
            # But for now, if stuck, we'll just stop and let local search fix it
            # Actually, we need a valid chain. Let's try a different strategy:
            # Grow in a spiral or use a known SAW generator
            pass
    
    # If the simple greedy failed, use a more robust method
    if len(chain) < n:
        # Fallback: generate a random SAW with rejection
        max_attempts = 1000
        for attempt in range(max_attempts):
            chain = [(0, 0, 0)]
            occupied = {(0, 0, 0)}
            success = True
            for step in range(1, n):
                last = chain[-1]
                dirs = list(directions)
                random.shuffle(dirs)
                found = False
                for d in dirs:
                    new_p = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
                    if new_p not in occupied:
                        chain.append(new_p)
                        occupied.add(new_p)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                break
        
        if not success:
            # Last resort: straight line (may not be optimal but is valid)
            chain = [(i, 0, 0) for i in range(n)]
    
    return chain

def _simulated_annealing(sequence, chain, time_limit, start_temp=5.0, cool_rate=0.9995):
    """Simulated annealing with pivot moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    temp = start_temp
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random pivot index
        i = random.randint(1, n - 2)
        axis = random.randint(0, 2)
        angle = random.choice([1, 2, 3])
        
        new_chain = _pivot_move(current_chain, i, axis, angle)
        if new_chain is None:
            continue
        
        # Check validity: only need to check if tail intersects head
        # Head is chain[0..i], tail is chain[i+1..n-1]
        # The tail is rigidly rotated, so internal self-avoidance is preserved
        # We only need to check tail vs head
        head_set = set(current_chain[:i+1])
        valid = True
        for p in new_chain[i+1:]:
            if p in head_set:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temp *= cool_rate
    
    return best_chain

def _multi_start_sa(sequence, dimension, time_per_sequence):
    """Run multiple starts of simulated annealing and return the best result."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Number of restarts based on time available
    # Each restart gets a fraction of the time
    # For short chains, more restarts; for long chains, fewer
    max_restarts = 5
    time_per_restart = time_per_sequence / max_restarts
    
    for restart in range(max_restarts):
        # Generate initial conformation
        if restart == 0:
            initial = _fast_grow(sequence, dimension)
        else:
            # Perturb the best found so far, or generate new
            if best_chain is not None and random.random() < 0.5:
                # Start from best, do a few random moves
                initial = list(best_chain)
                for _ in range(10):
                    i = random.randint(1, n - 2)
                    axis = random.randint(0, 2)
                    angle = random.choice([1, 2, 3])
                    new_c = _pivot_move(initial, i, axis, angle)
                    if new_c is not None:
                        # Check validity
                        head_set = set(initial[:i+1])
                        valid = True
                        for p in new_c[i+1:]:
                            if p in head_set:
                                valid = False
                                break
                        if valid:
                            initial = new_c
            else:
                initial = _fast_grow(sequence, dimension)
        
        # Run SA
        result = _simulated_annealing(sequence, initial, time_per_restart)
        e = _energy(sequence, result)
        
        if best_chain is None or e < best_energy:
            best_energy = e
            best_chain = result
        
        # Check if we're running out of time
        elapsed = time.time() - start_time
        if elapsed > time_per_sequence * 0.8:
            break
    
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Time budget allocation
    # Total budget is ~240s for all 26 sequences.
    # We are called once per sequence.
    # Allocate ~8 seconds per sequence to be safe (26 * 8 = 208s).
    # But we need a global deadline guard.
    
    # Since we don't have a global clock, we'll use a per-call limit.
    # If this is called in sequence, we can estimate.
    # But to be safe, let's use a fixed time per call.
    
    time_per_sequence = 8.0
    
    # Run multi-start simulated annealing
    best_chain = _multi_start_sa(sequence, dimension, time_per_sequence)
    
    if best_chain is None:
        # Fallback
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
