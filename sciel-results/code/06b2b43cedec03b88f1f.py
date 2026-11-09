"""
HP lattice protein folding on the cubic lattice.
Strategy: Multi-start simulated annealing with time-budgeted search.
"""
import random
import time
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 250.0  # Safety margin below 330s wall-clock


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _greedy_grow(sequence, dimension, rng, bias=0.5):
    """Randomized greedy chain growth with H-H contact preference."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
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
            score = gain * (1.0 + bias) + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain


def _compute_energy(sequence, chain):
    """Compute H-H contact energy (negative number of non-adjacent H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _pivot_move(chain, dimension, rng):
    """Perform a pivot move: rotate a segment around a bond."""
    n = len(chain)
    if n < 3:
        return None
    
    # Choose a pivot point (bond between i-1 and i)
    i = rng.randint(1, n - 2)
    
    # Get the segment to rotate
    segment = chain[i:]
    
    # Try all 6 possible rotations of the first bond in the segment
    d = [segment[0][k] - chain[i-1][k] for k in range(dimension)]
    
    # Generate all 6 rotations of vector d
    rotations = []
    for axis in range(dimension):
        for sign in (-1, 1):
            new_d = list(d)
            # Rotate around axis: swap two other axes
            other_axes = [a for a in range(dimension) if a != axis]
            a1, a2 = other_axes[0], other_axes[1]
            new_d[a1], new_d[a2] = new_d[a2] * sign, -new_d[a1] * sign
            rotations.append(tuple(new_d))
    
    # Try each rotation
    for new_d in rotations:
        # Build new segment
        new_segment = [tuple(chain[i-1][k] + new_d[k] for k in range(dimension))]
        for j in range(1, len(segment)):
            rel = [segment[j][k] - segment[j-1][k] for k in range(dimension)]
            prev = new_segment[-1]
            new_segment.append(tuple(prev[k] + rel[k] for k in range(dimension)))
        
        # Check if new segment is valid (no self-avoidance violations)
        valid = True
        new_chain = chain[:i] + new_segment
        for pos in new_segment:
            if pos in set(chain[:i]):
                valid = False
                break
            # Check within new segment
            if pos in new_segment[:-1]:
                valid = False
                break
        if not valid:
            continue
        
        return new_chain
    
    return None


def _slide_move(chain, dimension, rng):
    """Perform a slide move: move a single residue to an adjacent free site."""
    n = len(chain)
    if n < 2:
        return None
    
    i = rng.randint(1, n - 1)  # Don't move endpoints
    
    for site in _neighbours(chain[i]):
        if site in set(chain):
            continue
        
        new_chain = chain.copy()
        new_chain[i] = site
        return new_chain
    
    return None


def _simulated_annealing(sequence, chain, dimension, rng, time_budget):
    """Run simulated annealing on a given chain."""
    current = chain[:]
    current_energy = _compute_energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    
    start_time = time.time()
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_budget:
        # Choose move type
        if rng.random() < 0.7:
            new_chain = _pivot_move(current, dimension, rng)
        else:
            new_chain = _slide_move(current, dimension, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy
        
        T *= cooling_rate
    
    return best, best_energy


def fold(sequence, dimension):
    """Find the best conformation using multi-start simulated annealing."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1  # Energy is negative, so start high
    
    # Generate initial conformations
    initial_chains = []
    for _ in range(30):
        chain = _greedy_grow(sequence, dimension, rng, bias=rng.uniform(0.3, 1.0))
        if chain is not None:
            initial_chains.append(chain)
    
    # If no valid initial chain, use a straight line
    if not initial_chains:
        initial_chains = [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Run simulated annealing on each initial conformation
    elapsed = time.time() - _START_TIME
    remaining = _TOTAL_BUDGET - elapsed
    if remaining <= 0:
        # Return best found so far or a straight line
        if best_chain is None:
            best_chain = initial_chains[0]
        return [list(p) for p in best_chain]
    
    # Distribute time among restarts
    n_restarts = min(len(initial_chains), max(1, int(remaining / 2.0)))
    time_per_restart = remaining / n_restarts
    
    for i in range(n_restarts):
        if time.time() - _START_TIME > _TOTAL_BUDGET - 5.0:
            break
        
        chain = initial_chains[i]
        best, energy = _simulated_annealing(sequence, chain, dimension, rng, time_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = best
    
    if best_chain is None:
        best_chain = initial_chains[0]
    
    return [list(p) for p in best_chain]
