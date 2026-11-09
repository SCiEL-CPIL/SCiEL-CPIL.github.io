"""HP lattice protein folding using simulated annealing with multiple move types."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
DEADLINE = 195.0  # Safety margin before 210s kill

def _remaining_time():
    return DEADLINE - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _compute_energy_fast(sequence, chain):
    """Compute energy using a set for O(1) lookups."""
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _local_move(chain, rng):
    """Try to move a single residue to an adjacent empty site."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    old_pos = chain[i]
    occupied = set(chain)
    for new_pos in _neighbours(old_pos):
        if new_pos == chain[i - 1] or new_pos == chain[i + 1]:
            continue
        if new_pos in occupied:
            continue
        new_chain = chain.copy()
        new_chain[i] = new_pos
        return new_chain
    return None

def _reverse_move(chain, rng):
    """Reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None
    start = rng.randint(0, n - 4)
    end = rng.randint(start + 1, n - 1)
    new_chain = chain.copy()
    segment = new_chain[start:end + 1][::-1]
    new_chain[start:end + 1] = segment
    # Check for self-avoidance
    if len(set(new_chain)) != len(new_chain):
        return None
    return new_chain

def _rotate_move(chain, rng):
    """Rotate a segment around its first residue."""
    n = len(chain)
    if n < 5:
        return None
    start = rng.randint(0, n - 4)
    end = rng.randint(start + 2, n - 1)
    pivot = chain[start]
    segment = chain[start + 1:end + 1]
    
    # Try rotating the segment by 90 degrees around pivot
    rotated = []
    for pos in segment:
        dx = pos[0] - pivot[0]
        dy = pos[1] - pivot[1]
        # Rotate 90 degrees clockwise
        new_dx, new_dy = dy, -dx
        rotated.append((pivot[0] + new_dx, pivot[1] + new_dy))
    
    new_chain = chain.copy()
    new_chain[start + 1:end + 1] = rotated
    
    # Check for self-avoidance and valid steps
    if len(set(new_chain)) != len(new_chain):
        return None
    for i in range(len(new_chain) - 1):
        if abs(new_chain[i][0] - new_chain[i + 1][0]) + abs(new_chain[i][1] - new_chain[i + 1][1]) != 1:
            return None
    return new_chain

def _initial_chain(sequence, dimension, rng):
    """Create an initial random self-avoiding walk with H-preference."""
    n = len(sequence)
    if n <= 2:
        return [(0, 0), (1, 0)][:n]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            return None
        
        # Score candidates by potential H-H contacts
        if sequence[i] == 'H':
            scored = []
            for site in candidates:
                score = 0
                for j, pos in enumerate(chain):
                    if j < i and sequence[j] == 'H':
                        if abs(site[0] - pos[0]) + abs(site[1] - pos[1]) == 1:
                            score += 1
                scored.append((score, site))
            scored.sort(reverse=True)
            # Pick from top candidates with some randomness
            top_k = min(3, len(scored))
            _, site = scored[rng.randint(0, top_k - 1)]
        else:
            site = rng.choice(candidates)
        
        chain.append(site)
        occupied.add(site)
    
    return chain

def _anneal(sequence, dimension, rng, time_budget):
    """Run simulated annealing on a chain with improved move selection."""
    chain = _initial_chain(sequence, dimension, rng)
    if chain is None:
        return None, 0
    
    current_energy = _compute_energy_fast(sequence, chain)
    best_chain = chain.copy()
    best_energy = current_energy
    
    start_temp = 2.0
    end_temp = 0.05
    total_steps = 20000
    steps_per_second = 8000
    
    max_time = time_budget
    start_time = time.time()
    
    # Adaptive move type selection
    move_success = [0, 0, 0]
    move_total = [0, 0, 0]
    
    for step in range(total_steps):
        elapsed = time.time() - start_time
        if elapsed > max_time or _remaining_time() < 5.0:
            break
        
        progress = step / total_steps
        temperature = start_temp * (end_temp / start_temp) ** progress
        
        # Choose a move type with adaptive weights
        move_type = rng.randint(0, 2)
        new_chain = None
        
        if move_type == 0:
            new_chain = _local_move(chain, rng)
        elif move_type == 1:
            new_chain = _reverse_move(chain, rng)
        else:
            new_chain = _rotate_move(chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _compute_energy_fast(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.001)):
            chain = new_chain
            current_energy = new_energy
            move_success[move_type] += 1
            move_total[move_type] += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain.copy()
        else:
            move_total[move_type] += 1
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    best_chain, best_energy = None, 0
    
    # Allocate time budget based on remaining time
    remaining = _remaining_time()
    
    # Run multiple annealing attempts with decreasing time per attempt
    num_attempts = min(8, max(2, int(remaining / 15)))
    time_per_run = min(10.0, remaining / (num_attempts + 2))
    
    for attempt in range(num_attempts):
        if _remaining_time() < 8.0:
            break
        
        chain, energy = _anneal(sequence, dimension, rng, time_per_run)
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: simple greedy growth
        chain = _initial_chain(sequence, dimension, rng)
        if chain is None:
            best_chain = [(i, 0) for i in range(len(sequence))]
        else:
            best_chain = chain
            best_energy = _compute_energy_fast(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
