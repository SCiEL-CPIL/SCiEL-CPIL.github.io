"""HP lattice protein folding via simulated annealing with pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_TOTAL_BUDGET = 235.0
_start_time = None

def _time_remaining():
    if _start_time is None:
        return _TOTAL_BUDGET
    return max(0.0, _TOTAL_BUDGET - (time.time() - _start_time))

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _is_valid_walk(chain):
    """Check if chain is self-avoiding with unit steps."""
    if len(chain) != len(set(chain)):
        return False
    for i in range(1, len(chain)):
        if sum(abs(a - b) for a, b in zip(chain[i], chain[i-1])) != 1:
            return False
    return True

def _initial_snake(sequence, dimension, rng):
    """Generate a compact snake-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    # Try to fit in a box
    best = None
    for L in range(1, n + 1):
        for W in range(1, n + 1):
            D = max(1, (n + L * W - 1) // (L * W))
            if L * W * D < n:
                continue
            chain = []
            occupied = set()
            for z in range(D):
                for y in range(W):
                    xs = range(L) if z % 2 == 0 else range(L - 1, -1, -1)
                    for x in xs:
                        if len(chain) >= n:
                            break
                        p = (x, y, z)
                        if p not in occupied:
                            chain.append(p)
                            occupied.add(p)
                    if len(chain) >= n:
                        break
                if len(chain) >= n:
                    break
            if len(chain) < n:
                continue
            chain = chain[:n]
            if _is_valid_walk(chain):
                if best is None or len(occupied) < len(set(best)):
                    best = chain
    if best is not None:
        return best
    return [(i, 0, 0) for i in range(n)]

def _pivot_move_3d(chain, k, rng):
    """
    Rotate the tail chain[k:] around chain[k] by 90 or 180 degrees around a random axis.
    """
    n = len(chain)
    if k < 1 or k >= n - 1:
        return None
    
    pivot = chain[k]
    tail = chain[k:]
    
    # Pick a random axis and angle
    axis = rng.randint(0, 2)
    angle = rng.choice([90, -90, 180])
    
    new_tail = [pivot]
    for i in range(1, len(tail)):
        p = tail[i]
        rx = p[0] - pivot[0]
        ry = p[1] - pivot[1]
        rz = p[2] - pivot[2]
        
        if axis == 0:
            if angle == 90:
                nx, ny, nz = rx, -rz, ry
            elif angle == -90:
                nx, ny, nz = rx, rz, -ry
            else:
                nx, ny, nz = rx, -ry, -rz
        elif axis == 1:
            if angle == 90:
                nx, ny, nz = rz, ry, -rx
            elif angle == -90:
                nx, ny, nz = -rz, ry, rx
            else:
                nx, ny, nz = -rx, ry, -rz
        else:
            if angle == 90:
                nx, ny, nz = -ry, rx, rz
            elif angle == -90:
                nx, ny, nz = ry, -rx, rz
            else:
                nx, ny, nz = -rx, -ry, rz
        
        new_tail.append((pivot[0] + nx, pivot[1] + ny, pivot[2] + nz))
    
    new_chain = chain[:k] + new_tail
    if _is_valid_walk(new_chain):
        return new_chain
    return None

def _anneal(sequence, chain, rng, time_limit):
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    T = max(1.0, -best_energy * 0.3)
    T_min = 0.05
    cooling = 0.997
    start_t = time.time()
    
    while time.time() - start_t < time_limit and T > T_min:
        k = rng.randint(1, n - 2)
        new_chain = _pivot_move_3d(current_chain, k, rng)
        
        if new_chain is None:
            T *= cooling
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        T *= cooling
    
    return best_chain, best_energy

def fold(sequence, dimension):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    remaining = _time_remaining()
    # Allocate time proportional to sequence length
    seq_time = max(0.3, min(remaining * 0.3, (n / 50.0) * 5.0))
    
    rng = random.Random(123 + n)
    best_chain = None
    best_energy = 0
    
    # Multiple restarts
    num_restarts = max(2, min(6, int(seq_time / 1.5) + 1))
    per_restart_time = seq_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() - _start_time > _TOTAL_BUDGET - 5:
            break
        
        init = _initial_snake(sequence, dimension, rng)
        # Perturb initial conformation for diversity
        if r > 0:
            for _ in range(3):
                k = rng.randint(1, n - 2)
                nc = _pivot_move_3d(init, k, rng)
                if nc is not None:
                    init = nc
        
        chain, energy = _anneal(sequence, init, rng, per_restart_time)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
