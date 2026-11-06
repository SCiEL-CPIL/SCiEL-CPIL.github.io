"""HP lattice protein folding on the square lattice: Simulated Annealing with Pivot Moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s limit, 15s margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbors(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def _energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == "H"]
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for nx, ny in _neighbors((x, y)):
            j = pos_to_idx.get((nx, ny))
            if j is not None and abs(i - j) > 1:
                if sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow_initial(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using random growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Try to generate a compact walk
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for nb in _neighbors(last):
            if nb not in occupied:
                options.append(nb)
                
        if not options:
            # Failed to grow, fallback to straight line
            return [(i, 0) for i in range(n)]
            
        # Heuristic: prefer moves that keep the chain compact or allow future growth
        # Simple random choice is often sufficient with SA
        next_pos = rng.choice(options)
        chain.append(next_pos)
        occupied[next_pos] = i
        
    return chain


def _rotate_vec(v, angle):
    x, y = v
    if angle == 90:
        return (-y, x)
    elif angle == -90:
        return (y, -x)
    elif angle == 180:
        return (-x, -y)
    else:
        return (x, y)


def _pivot_move(chain, k, angle):
    """
    Attempt a pivot move at index k with given angle.
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if k <= 0 or k >= n - 1:
        return None
    
    pivot_point = chain[k-1]
    
    # We construct the new chain
    new_chain = chain[:k]  # Copy prefix 0..k-1
    
    # Calculate new positions for suffix k..n-1
    for i in range(k, n):
        vec = (chain[i][0] - pivot_point[0], chain[i][1] - pivot_point[1])
        new_vec = _rotate_vec(vec, angle)
        new_pos = (pivot_point[0] + new_vec[0], pivot_point[1] + new_vec[1])
        new_chain.append(new_pos)
    
    # Check self-avoidance against the fixed prefix (0..k-2)
    fixed_set = set(chain[:k-1])
    
    for i in range(k, n):
        if new_chain[i] in fixed_set:
            return None
            
    return new_chain


def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + n)
    
    # 1. Generate Initial Chain
    current_chain = _grow_initial(sequence, dimension, rng)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = current_chain
    best_energy = current_energy
    
    # 2. Simulated Annealing
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    T = T_start
    
    while _remaining() > 1.0:
        # Pick a random pivot index and angle
        k = rng.randint(1, n - 2)
        angle = rng.choice([90, -90, 180])
        
        new_chain = _pivot_move(current_chain, k, angle)
        
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain
                
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
