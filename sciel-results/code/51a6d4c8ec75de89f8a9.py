"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 170.0  # Safety margin for 210s kill (evaluator says 180s per call, but 210s wall-clock kill)


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


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
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Try a pivot move: rotate a subchain. Returns new chain or None if invalid."""
    n = len(chain)
    if n < 4:
        return None
    # Choose pivot point
    pivot = rng.randint(2, n - 2)
    # Choose rotation: in 2D, we can rotate the tail by 90 or 180 degrees
    # Actually, pivot moves in 2D: rotate the subchain from pivot to end by 90 or 270 degrees
    # Let's implement a simpler version: try moving a random internal segment
    
    # Alternative: crankshaft move - pick a segment and shift it
    # Let's do a proper pivot: pick pivot index, rotate the tail
    head = chain[:pivot]
    tail = chain[pivot:]
    
    # The tail starts at chain[pivot], which is connected to chain[pivot-1]
    # We rotate the tail around chain[pivot]
    # In 2D, rotation by 90 degrees: (x,y) -> (-y,x) or (y,-x) relative to pivot
    
    px, py = chain[pivot]
    
    # Try rotation by 90 degrees clockwise: (dx,dy) -> (dy,-dx)
    new_tail = [chain[pivot]]
    for i in range(1, len(tail)):
        dx = tail[i][0] - tail[i-1][0]
        dy = tail[i][1] - tail[i-1][1]
        # Rotate 90 deg clockwise: (dx,dy) -> (dy,-dx)
        new_dx = dy
        new_dy = -dx
        new_tail.append((new_tail[-1][0] + new_dx, new_tail[-1][1] + new_dy))
    
    new_chain = head + new_tail
    
    # Check self-avoidance
    if len(set(new_chain)) != n:
        return None
    
    # Check that the connection at pivot is valid (unit step)
    # head[-1] to new_tail[0] should be unit step
    dx = new_tail[0][0] - head[-1][0]
    dy = new_tail[0][1] - head[-1][1]
    if abs(dx) + abs(dy) != 1:
        return None
    
    return new_chain


def _local_search(sequence, chain, rng, max_steps=500):
    """Local search using pivot moves and energy minimization."""
    current = chain[:]
    current_energy = _energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        # Try a pivot move
        new_chain = _pivot_move(current, rng)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        # Accept if better, or with some probability if worse (simulated annealing-like)
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy
        else:
            # Accept worse moves with decreasing probability
            delta = current_energy - new_energy  # positive if worse
            accept_prob = math.exp(-delta / max(1, 100.0 / (step + 1)))
            if rng.random() < accept_prob:
                current = new_chain
                current_energy = new_energy
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized growth + local search with time budget."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    best = None
    best_energy = 1  # Energy is negative, so 1 is "worse" than any valid fold
    
    # Phase 1: Generate initial solutions via greedy growth
    num_growths = 100
    for i in range(num_growths):
        if _time_left() < 10.0:
            break
        
        # Vary the bias for exploration
        bias = 0.5 + 0.5 * (i % 10) / 10.0
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best = chain[:]
            best_energy = energy
    
    # If no valid fold found, use straight line
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)
    
    # Phase 2: Local search from the best solution
    if _time_left() > 15.0:
        improved, improved_energy = _local_search(sequence, best, rng, max_steps=2000)
        if improved_energy < best_energy:
            best = improved
            best_energy = improved_energy
    
    # Phase 3: Try more growths with different biases if time permits
    if _time_left() > 20.0:
        for i in range(200):
            if _time_left() < 10.0:
                break
            
            bias = 0.3 + 0.7 * rng.random()
            chain = _grow(sequence, dimension, rng, bias=bias)
            if chain is None:
                continue
            
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best = chain[:]
                best_energy = energy
                
                # Quick local search from this new best
                if _time_left() > 15.0:
                    improved, improved_energy = _local_search(sequence, best, rng, max_steps=500)
                    if improved_energy < best_energy:
                        best = improved
                        best_energy = improved_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
