"""HP lattice protein folding on the square lattice using pivot-based simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)

def _neighbours(point):
    """Generate 4-neighbours of a point on the square lattice."""
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
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

def _greedy_grow(sequence, rng):
    """Generate an initial self-avoiding walk using greedy growth with H-contact bias."""
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        x, y = chain[-1]
        candidates = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if pos in occupied:
                continue
            
            score = 0.0
            if sequence[i] == 'H':
                for px, py in [(nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)]:
                    p = (px, py)
                    if p in occupied:
                        idx = occupied[p]
                        if idx < i - 1 and sequence[idx] == 'H':
                            score += 1.0
            
            # Add random noise to break ties
            score += rng.random() * 0.05
            candidates.append((score, pos))
        
        if not candidates:
            # Dead end - fail
            return None
        
        # Pick best candidate
        best_score, best_pos = max(candidates)
        chain.append(best_pos)
        occupied[best_pos] = i
    
    return chain

def _rotate_around_center(point, center, angle):
    """Rotate point around center by angle (90, 180, 270 degrees)."""
    x, y = point
    cx, cy = center
    dx = x - cx
    dy = y - cy
    
    if angle == 90:
        return (cx - dy, cy + dx)
    elif angle == 180:
        return (cx - dx, cy - dy)
    elif angle == 270:
        return (cx + dy, cy - dx)
    else:
        return (x, y)

def _try_pivot_move(chain, pivot_idx, angle, rng):
    """Attempt a pivot move. Returns new chain if valid, else None."""
    n = len(chain)
    center = chain[pivot_idx]
    head = chain[:pivot_idx + 1]
    tail = chain[pivot_idx + 1:]
    
    if len(tail) == 0:
        return None
    
    new_tail = [_rotate_around_center(p, center, angle) for p in tail]
    new_chain = head + new_tail
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    return new_chain

def _pivot_search(sequence, initial_chain, time_limit, rng):
    """Improve the conformation using pivot-based simulated annealing."""
    if not initial_chain:
        return None, 0
    
    current_chain = initial_chain
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    if n < 4:
        return best_chain, best_energy
    
    start_time = time.time()
    
    # Simulated annealing parameters
    T_start = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    T = T_start
    stall_count = 0
    max_stall = 3000
    
    while True:
        if time.time() - start_time > time_limit:
            break
        if _remaining_time() < 5.0:
            break
        
        # Pick a random pivot index
        # Tail must have at least 2 residues for meaningful rotation
        pivot_idx = rng.randint(1, n - 3)
        angle = rng.choice([90, 180, 270])
        
        new_chain = _try_pivot_move(current_chain, pivot_idx, angle, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e < 0:
            # Better energy, always accept
            current_chain = new_chain
            current_energy = new_energy
            stall_count = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    current_chain = new_chain
                    current_energy = new_energy
            else:
                # At low temperature, only accept if it's a kick
                if rng.random() < 0.005:
                    current_chain = new_chain
                    current_energy = new_energy
            
            stall_count += 1
            if stall_count > max_stall:
                # Reset stall and cool down
                stall_count = 0
                if T > T_min:
                    T *= cooling_rate
                    max_stall = int(max_stall * 0.8)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    if _remaining_time() < 2.0:
        # Out of time, return straight line
        return [[i, 0] for i in range(n)]
    
    # Allocate time based on remaining budget
    # If we have plenty of time, use more for this sequence
    remaining = _remaining_time()
    if remaining > 100.0:
        time_limit = min(15.0, remaining - 5.0)
    elif remaining > 50.0:
        time_limit = min(12.0, remaining - 5.0)
    elif remaining > 20.0:
        time_limit = min(8.0, remaining - 5.0)
    else:
        time_limit = max(1.0, remaining - 2.0)
    
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Try multiple initial conformations with different seeds
    num_initials = 3
    
    for k in range(num_initials):
        if _remaining_time() < 3.0:
            break
        
        # Generate initial conformation
        init_rng = random.Random(12345 + n + k * 1000)
        initial = _greedy_grow(sequence, init_rng)
        
        if initial is None:
            # Fallback: straight line
            initial = [(i, 0) for i in range(n)]
        
        # Refine using pivot search
        refined_chain, refined_energy = _pivot_search(sequence, initial, time_limit / (num_initials - k + 1), rng)
        
        if refined_chain is not None:
            if best_chain is None or refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
