"""HP lattice protein folding using simulated annealing on self-avoiding walks."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute energy from a chain: -1 for each non-adjacent H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk (spiral or random growth)."""
    n = len(sequence)
    # Try a simple spiral first for better initial packing
    chain = []
    occupied = set()
    # Spiral pattern for square lattice
    if n <= 1:
        return [(0, 0)]
    
    # Generate a spiral
    x, y = 0, 0
    chain.append((x, y))
    occupied.add((x, y))
    
    dx, dy = 1, 0
    step_count = 1
    steps_in_direction = 1
    direction_changes = 0
    
    for i in range(1, n):
        # Move in current direction
        x += dx
        y += dy
        if (x, y) in occupied or len(chain) >= n:
            # Try to turn
            break
        chain.append((x, y))
        occupied.add((x, y))
        step_count += 1
        if step_count >= steps_in_direction:
            step_count = 0
            direction_changes += 1
            # Rotate direction
            dx, dy = -dy, dx
            if direction_changes % 2 == 0:
                steps_in_direction += 1
        if len(chain) >= n:
            break
    
    if len(chain) < n:
        # Spiral failed, use random growth
        chain = [(0, 0)]
        occupied = {(0, 0)}
        for i in range(1, n):
            last = chain[-1]
            options = [q for q in _neighbours(last) if q not in occupied]
            if not options:
                # Dead end, try to backtrack or just stop
                break
            site = rng.choice(options)
            chain.append(site)
            occupied.add(site)
        if len(chain) < n:
            # Fallback: straight line
            chain = [(i, 0) for i in range(n)]
    
    return chain[:n]


def _is_valid_chain(chain, dimension):
    """Check if chain is a valid self-avoiding walk with unit steps."""
    if len(chain) == 0:
        return True
    occupied = set()
    for i, p in enumerate(chain):
        if p in occupied:
            return False
        occupied.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = sum(abs(a-b) for a, b in zip(p, prev))
            if dist != 1:
                return False
    return True


def _pivot_move(chain, rng):
    """Pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 3:
        return None
    
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    prefix = chain[:pivot_idx]
    suffix = chain[pivot_idx:]
    
    # Try all 4 rotations of the suffix
    for _ in range(4):
        # Rotate suffix around pivot
        new_suffix = [pivot]
        for i in range(1, len(suffix)):
            x, y = suffix[i]
            # Rotate 90 degrees clockwise around pivot
            new_x = pivot[0] + (y - pivot[1])
            new_y = pivot[1] - (x - pivot[0])
            new_suffix.append((new_x, new_y))
        
        new_chain = prefix + new_suffix[1:]
        if _is_valid_chain(new_chain, 2):
            return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Crankshaft move: rotate a middle segment of the chain."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length
    seg_len = rng.randint(2, min(4, n - 2))
    start_idx = rng.randint(1, n - seg_len - 1)
    
    prefix = chain[:start_idx]
    segment = chain[start_idx:start_idx + seg_len]
    suffix = chain[start_idx + seg_len:]
    
    # Try rotating the segment
    for _ in range(4):
        # Rotate segment around its first point
        anchor = segment[0]
        new_segment = [anchor]
        for i in range(1, len(segment)):
            x, y = segment[i]
            new_x = anchor[0] + (y - anchor[1])
            new_y = anchor[1] - (x - anchor[0])
            new_segment.append((new_x, new_y))
        
        new_chain = prefix + new_segment + suffix
        if _is_valid_chain(new_chain, 2):
            return new_chain
    
    return None


def _single_site_move(chain, rng):
    """Try to move a single residue to an adjacent free site."""
    n = len(chain)
    if n < 3:
        return None
    
    idx = rng.randint(1, n - 2)
    site = chain[idx]
    
    for new_site in _neighbours(site):
        if new_site in chain:
            continue
        # Check if moving would create a collision
        new_chain = chain[:]
        new_chain[idx] = new_site
        if _is_valid_chain(new_chain, 2):
            return new_chain
    
    return None


def _local_search(sequence, chain, max_iters, rng, deadline):
    """Simulated annealing local search."""
    current = list(chain)
    current_energy = _compute_energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    n = len(sequence)
    T = 5.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.3:
            new_chain = _pivot_move(current, rng)
        elif move_type < 0.6:
            new_chain = _crankshaft_move(current, rng)
        else:
            new_chain = _single_site_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Acceptance criterion
        if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best = list(current)
                best_energy = current_energy
        
        T *= cooling
    
    return best, best_energy


def fold(sequence, dimension):
    """Fold a sequence using simulated annealing on self-avoiding walks."""
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    # Determine search budget based on sequence length
    if n <= 30:
        max_restarts = 8
        sa_iters = 2000
    elif n <= 50:
        max_restarts = 5
        sa_iters = 1500
    elif n <= 80:
        max_restarts = 3
        sa_iters = 1000
    else:
        max_restarts = 2
        sa_iters = 800
    
    best_chain = None
    best_energy = 0  # Energy is negative, so start at 0 (worst possible)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:  # Leave 5s safety margin
            break
        
        # Generate initial chain
        initial = _initial_chain(sequence, dimension, rng)
        if len(initial) < n:
            initial = _initial_chain(sequence, dimension, rng)
            if len(initial) < n:
                initial = [(i, 0) for i in range(n)]
        
        # Run SA
        chain, energy = _local_search(sequence, initial, sa_iters, rng, deadline)
        
        if energy < best_energy or best_chain is None:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
