"""
HP lattice protein folding on the cubic lattice.
Strategy: Multi-start with simulated annealing using pivot and slide moves.
Time-budgeted global search across all sequences.
"""
import random
import time
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 280.0  # Safety margin before 330s wall-clock limit


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


def _greedy_grow(sequence, dimension, rng):
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _spiral_seed(sequence, dimension, rng):
    """Generate a spiral-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    # Try to create a compact spiral pattern
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    direction = [(1, 0, 0)]  # Start moving in +x
    step_count = 0
    steps_in_direction = 1
    next_turn = 1
    
    for i in range(1, n):
        # Try to continue in current direction
        candidate = (chain[-1][0] + direction[0], 
                     chain[-1][1] + direction[1], 
                     chain[-1][2] + direction[2])
        
        if candidate not in occupied:
            chain.append(candidate)
            occupied[candidate] = i
            step_count += 1
            steps_in_direction += 1
            
            if steps_in_direction >= next_turn:
                # Turn 90 degrees (rotate direction)
                d = direction
                # Rotate in xy plane: (dx, dy, dz) -> (-dy, dx, dz)
                direction = (-d[1], d[0], d[2])
                steps_in_direction = 1
                next_turn += 2
        else:
            # Try rotating
            d = direction
            direction = (-d[1], d[0], d[2])
            steps_in_direction = 1
            next_turn += 2
    
    return chain


def _pivot_move(chain, rng):
    """Apply a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 3:
        return None
    
    # Choose a pivot index (not the first two, not the last)
    pivot_idx = rng.randint(2, n - 2)
    pivot = chain[pivot_idx]
    
    # Choose a random rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.choice([90, -90, 180])
    
    # Define rotation matrices for 90, -90, 180 degrees around each axis
    def rotate_point(p, pivot, axis, angle):
        rel = [p[i] - pivot[i] for i in range(3)]
        
        if angle == 180:
            # 180 degree rotation: flip the two non-axis coordinates
            if axis == 0:
                rel[1] = -rel[1]
                rel[2] = -rel[2]
            elif axis == 1:
                rel[0] = -rel[0]
                rel[2] = -rel[2]
            else:
                rel[0] = -rel[0]
                rel[1] = -rel[1]
        elif angle == 90:
            # 90 degree rotation around axis
            if axis == 0:
                new_rel = [rel[0], -rel[2], rel[1]]
            elif axis == 1:
                new_rel = [rel[2], rel[1], -rel[0]]
            else:
                new_rel = [-rel[1], rel[0], rel[2]]
            rel = new_rel
        elif angle == -90:
            # -90 degree rotation around axis
            if axis == 0:
                new_rel = [rel[0], rel[2], -rel[1]]
            elif axis == 1:
                new_rel = [-rel[2], rel[1], rel[0]]
            else:
                new_rel = [rel[1], -rel[0], rel[2]]
            rel = new_rel
        
        return tuple(pivot[i] + rel[i] for i in range(3))
    
    # Apply rotation to suffix
    new_chain = chain[:pivot_idx]
    for i in range(pivot_idx, n):
        new_chain.append(rotate_point(chain[i], pivot, axis, angle))
    
    # Check self-avoidance
    seen = set()
    for i, p in enumerate(new_chain):
        if p in seen:
            return None
        seen.add(p)
    
    # Check unit steps
    for i in range(len(new_chain) - 1):
        diff = abs(new_chain[i+1][0] - new_chain[i][0]) + \
               abs(new_chain[i+1][1] - new_chain[i][1]) + \
               abs(new_chain[i+1][2] - new_chain[i][2])
        if diff != 1:
            return None
    
    return new_chain


def _slide_move(chain, rng):
    """Apply a slide move: move one residue to an adjacent free site."""
    n = len(chain)
    if n < 2:
        return None
    
    # Choose a residue index (not the first or last)
    idx = rng.randint(1, n - 2)
    point = chain[idx]
    
    # Try to move to a random neighboring site
    neighbours = list(_neighbours(point))
    rng.shuffle(neighbours)
    
    for new_point in neighbours:
        # Check if new_point is free (not occupied by other residues)
        if new_point in chain:
            continue
        
        # Create new chain
        new_chain = chain[:]
        new_chain[idx] = new_point
        
        # Check self-avoidance
        seen = set()
        for i, p in enumerate(new_chain):
            if p in seen:
                break
            seen.add(p)
        else:
            # Check unit steps
            valid = True
            for i in range(len(new_chain) - 1):
                diff = abs(new_chain[i+1][0] - new_chain[i][0]) + \
                       abs(new_chain[i+1][1] - new_chain[i][1]) + \
                       abs(new_chain[i+1][2] - new_chain[i][2])
                if diff != 1:
                    valid = False
                    break
            
            if valid:
                return new_chain
    
    return None


def _simulated_annealing(sequence, chain, rng, time_limit):
    """Run simulated annealing on a given chain."""
    n = len(sequence)
    best_chain = chain[:]
    best_energy = _energy(sequence, chain)
    current_chain = chain[:]
    current_energy = best_energy
    
    # Initial temperature based on typical energy scale
    t0 = max(1.0, abs(best_energy) * 0.5)
    t = t0
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Choose move type
        if rng.random() < 0.7:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _slide_move(current_chain, rng)
        
        if new_chain is None:
            t *= cooling_rate
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / t) if worse
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(t, 1e-10)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        
        t *= cooling_rate
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multi-start simulated annealing with time budget."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Generate initial conformations
    initial_chains = []
    
    # Try greedy growth multiple times
    for _ in range(10):
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    # Try spiral seed
    spiral = _spiral_seed(sequence, dimension, rng)
    if spiral is not None and len(spiral) == n:
        initial_chains.append(spiral)
    
    # Add a simple random walk as fallback
    def _random_walk():
        chain = [(0, 0, 0)]
        occupied = {(0, 0, 0): 0}
        for i in range(1, n):
            neighbours = list(_neighbours(chain[-1]))
            rng.shuffle(neighbours)
            for site in neighbours:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    break
            else:
                return None
        return chain
    
    rw = _random_walk()
    if rw is not None:
        initial_chains.append(rw)
    
    if not initial_chains:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Check time remaining
    elapsed = time.time() - _START_TIME
    remaining = _TOTAL_BUDGET - elapsed
    if remaining <= 5:
        # Not enough time, return best initial
        best = min(initial_chains, key=lambda c: _energy(sequence, c))
        return [list(p) for p in best]
    
    # Allocate time per initial chain
    num_starts = min(len(initial_chains), 5)  # Limit to 5 starts
    time_per_start = max(1.0, (remaining - 5) / num_starts)
    
    best_chain = None
    best_energy = 1
    
    for i in range(num_starts):
        chain = initial_chains[i]
        sa_chain, sa_energy = _simulated_annealing(sequence, chain, rng, time_per_start)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    if best_chain is None:
        best_chain = initial_chains[0]
    
    return [list(p) for p in best_chain]
