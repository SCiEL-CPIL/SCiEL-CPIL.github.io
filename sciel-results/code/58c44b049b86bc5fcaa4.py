"""Rewritten HP lattice protein folding using simulated annealing with segment-reversal moves."""
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_spiral(sequence, dimension):
    """Generate an initial compact spiral-like conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Start with a simple spiral pattern
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Directions: right, down, left, up
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_in_dir = 1
    steps_taken = 0
    
    for i in range(1, n):
        placed = False
        while not placed:
            dx, dy = directions[dir_idx]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                placed = True
                steps_taken += 1
                if steps_taken == steps_in_dir:
                    steps_taken = 0
                    dir_idx = (dir_idx + 1) % 4
                    if dir_idx % 2 == 0:
                        steps_in_dir += 1
            else:
                dir_idx = (dir_idx + 1) % 4
                if dir_idx == 0:
                    steps_in_dir += 1
                # If we've gone full circle without placing, try a different approach
                if dir_idx == 0 and steps_in_dir > n:
                    # Fallback: just place linearly
                    for j in range(i, n):
                        pos = (chain[j-1][0] + 1, chain[j-1][1])
                        while pos in occupied:
                            pos = (pos[0], pos[1] + 1)
                        chain.append(pos)
                        occupied[pos] = j
                    return chain
        if len(chain) == n:
            break
    
    return chain


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid_walk(chain):
    """Check if the chain is a valid self-avoiding walk with unit steps."""
    if len(chain) == 0:
        return True
    if len(chain) == 1:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0] - prev[0]) + abs(p[1] - prev[1])
            if dist != 1:
                return False
    return True


def _reverse_segment(chain, start, end):
    """Reverse a segment of the chain and check validity."""
    if start >= end:
        return chain, False
    
    new_chain = chain[:start] + chain[start:end+1][::-1] + chain[end+1:]
    
    # Check self-avoidance
    occupied = set()
    for p in new_chain:
        if p in occupied:
            return chain, False
        occupied.add(p)
    
    # Check unit steps
    for i in range(1, len(new_chain)):
        dist = abs(new_chain[i][0] - new_chain[i-1][0]) + abs(new_chain[i][1] - new_chain[i-1][1])
        if dist != 1:
            return chain, False
    
    return new_chain, True


def _move_chain(chain, rng):
    """Perform a random segment reversal move."""
    n = len(chain)
    if n < 2:
        return chain
    
    # Choose a random segment to reverse
    # Prefer shorter segments for local moves, occasionally longer
    if rng.random() < 0.7:
        length = rng.randint(2, min(10, n))
    else:
        length = rng.randint(2, n)
    
    start = rng.randint(0, n - length)
    end = start + length - 1
    
    new_chain, valid = _reverse_segment(chain, start, end)
    return new_chain


def fold(sequence, dimension):
    """Best conformation using simulated annealing with segment-reversal moves."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s budget per sequence, leaving margin
    
    # Initial conformation
    chain = _initial_spiral(sequence, dimension)
    if not _is_valid_walk(chain):
        # Fallback to linear
        chain = [(i, 0) for i in range(n)]
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    rng = random.Random(12345 + len(sequence))
    
    while time.time() < deadline and T > T_min:
        # Generate a new conformation
        new_chain = _move_chain(chain, rng)
        
        if new_chain is chain:
            T *= cooling_rate
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling_rate
    
    # Additional local search with smaller temperature if time remains
    remaining = deadline - time.time()
    if remaining > 5:
        T = 0.5
        cooling_rate = 0.9995
        while time.time() < deadline - 2 and T > T_min:
            new_chain = _move_chain(chain, rng)
            
            if new_chain is chain:
                T *= cooling_rate
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            
            T *= cooling_rate
    
    # Ensure validity
    if not _is_valid_walk(best_chain):
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
