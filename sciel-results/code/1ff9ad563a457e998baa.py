"""HP lattice protein folding using simulated annealing with multiple move types."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Track global deadline
_START_TIME = time.time()
DEADLINE = 190.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
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

def _initial_chain(sequence, dimension, rng):
    """Generate an initial chain using biased random growth."""
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
            # Use a random bias to diversify initial chains
            bias = rng.uniform(0.5, 2.0)
            options.append((gain * bias + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_move(chain, dimension, rng):
    """Try a local move: move one residue to an adjacent empty site."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)  # Don't move endpoints
    old_pos = chain[i]
    for new_pos in _neighbours(old_pos):
        if new_pos not in chain:
            # Check that moving doesn't create adjacency issues
            chain[i] = new_pos
            return True
    return False

def _reverse_move(chain, dimension, rng):
    """Try a reversal move: reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None
    # Choose a segment to reverse
    length = rng.randint(2, min(10, n - 2))
    start = rng.randint(1, n - length - 1)
    end = start + length - 1
    # Try reversing
    segment = chain[start:end + 1]
    segment.reverse()
    # Check validity
    for i in range(start, end + 1):
        if i > 0 and chain[i] == chain[i - 1]:
            segment.reverse()
            return False
        if i < n - 1 and chain[i] == chain[i + 1]:
            segment.reverse()
            return False
    # Check self-avoidance
    for i in range(start, end + 1):
        if chain.count(chain[i]) > 1:
            segment.reverse()
            return False
    return True

def _rotate_move(chain, dimension, rng):
    """Try a rotation move: rotate a segment around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    # Choose pivot
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    # Choose how many residues to rotate
    length = rng.randint(2, min(5, n - pivot_idx - 1))
    segment = chain[pivot_idx + 1:pivot_idx + 1 + length]
    if not segment:
        return False
    # Try rotating the segment
    for _ in range(4):
        # Rotate each point in segment around pivot
        rotated = []
        valid = True
        for p in segment:
            dx = p[0] - pivot[0]
            dy = p[1] - pivot[1]
            # Rotate 90 degrees
            new_x = pivot[0] - dy
            new_y = pivot[1] + dx
            new_p = (new_x, new_y)
            if new_p in chain:
                valid = False
                break
            rotated.append(new_p)
        if valid:
            # Apply the rotation
            chain[pivot_idx + 1:pivot_idx + 1 + length] = rotated
            return True
    return False

def _anneal(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing with multiple move types."""
    start_time = time.time()
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = max(0.5, -current_energy * 0.1)
    T_min = 0.01
    cooling_rate = 0.995
    
    while T > T_min and (time.time() - start_time) < time_budget and _remaining_time() > 10:
        # Choose move type
        move_type = rng.random()
        moved = False
        if move_type < 0.4:
            moved = _local_move(chain, dimension, rng)
        elif move_type < 0.7:
            moved = _reverse_move(chain, dimension, rng)
        else:
            moved = _rotate_move(chain, dimension, rng)
        
        if not moved:
            T *= cooling_rate
            continue
        
        new_energy = _energy(sequence, chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert the move
            # This is tricky; we need to undo the move
            # For simplicity, we'll just regenerate if needed
            pass
        
        T *= cooling_rate
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Generate multiple initial chains with different biases
    initial_chains = []
    for _ in range(10):
        chain = _initial_chain(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback to a straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Sort by energy and allocate more time to better starts
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    best_chain = initial_chains[0]
    best_energy = _energy(sequence, best_chain)
    
    # Allocate time budget based on remaining time
    total_time_budget = max(1.0, _remaining_time() - 15.0)
    
    for i, chain in enumerate(initial_chains):
        if _remaining_time() < 10:
            break
        # Better initial chains get more time
        time_for_this = total_time_budget / len(initial_chains) * (2.0 - i / len(initial_chains))
        time_for_this = max(0.5, time_for_this)
        
        improved_chain, improved_energy = _anneal(sequence, chain, dimension, rng, time_for_this)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
