"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-start Monte Carlo with simulated annealing local optimization.
"""
import random
import time
import math

# EVOLVE-BLOCK-START

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

def _valid_move(chain, occupied, i, new_pos):
    """Check if moving residue i to new_pos is valid (self-avoiding, unit step)."""
    if new_pos in occupied and occupied[new_pos] != i:
        return False
    # Check unit step from predecessor
    if i > 0:
        pred = chain[i - 1]
        if sum(abs(a - b) for a, b in zip(new_pos, pred)) != 1:
            return False
    # Check unit step to successor
    if i < len(chain) - 1:
        succ = chain[i + 1]
        if sum(abs(a - b) for a, b in zip(new_pos, succ)) != 1:
            return False
    return True

def _local_search(sequence, chain, occupied, time_limit, rng):
    """Simulated annealing local search to improve a conformation."""
    n = len(sequence)
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    current_occupied = dict(occupied)
    
    start_time = time.time()
    temp = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    while temp > min_temp:
        if time.time() - start_time > time_limit:
            break
        
        # Try a random single-residue move
        if n <= 1:
            break
        i = rng.randint(1, n - 2)  # avoid endpoints for simplicity
        old_pos = current_chain[i]
        
        # Generate a random neighbor
        neighbors = list(_neighbours(old_pos))
        new_pos = rng.choice(neighbors)
        
        if not _valid_move(current_chain, current_occupied, i, new_pos):
            continue
        
        # Compute energy change
        # Remove old position, add new position
        delta = 0
        # Count H-H contacts lost/changed
        is_h = sequence[i] == "H"
        
        # Old contacts involving position i
        old_contacts = 0
        if is_h:
            for q in _neighbours(old_pos):
                j = current_occupied.get(q)
                if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                    old_contacts += 1
        
        # New contacts involving position i
        new_contacts = 0
        if is_h:
            for q in _neighbours(new_pos):
                j = current_occupied.get(q)
                if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                    new_contacts += 1
        
        delta = -(new_contacts - old_contacts)
        
        # Acceptance
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            # Perform the move
            del current_occupied[old_pos]
            current_occupied[new_pos] = i
            current_chain[i] = new_pos
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temp *= cooling
    
    return best_chain, best_energy

def _grow_greedy(sequence, dimension, rng, bias=0.3):
    """Randomized greedy growth with tunable bias toward contacts."""
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
            score = gain * bias + 0.1 * free + rng.random() * (1 - bias)
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _grow_random_walk(sequence, dimension, rng):
    """Random walk growth with backtracking on collision."""
    max_attempts = 50
    for _ in range(max_attempts):
        start = (0,) * dimension
        chain, occupied = [start], {start: 0}
        success = True
        for i in range(1, len(sequence)):
            neighbors = [s for s in _neighbours(chain[-1]) if s not in occupied]
            if not neighbors:
                success = False
                break
            site = rng.choice(neighbors)
            occupied[site] = i
            chain.append(site)
        if success:
            return chain
    return None

def fold(sequence, dimension):
    """Best conformation using multi-start Monte Carlo with local optimization."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    global_start = time.time()
    total_time_limit = 320.0  # Leave 10s margin before 330s kill
    per_sequence_time = total_time_limit / 26.0  # Rough budget per sequence
    # For longer sequences, give them more time
    if n > 64:
        per_sequence_time = min(per_sequence_time * 2, 60.0)
    else:
        per_sequence_time = min(per_sequence_time, 30.0)
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Generate diverse initial conformations
    initial_chains = []
    
    # Greedy growth with different biases
    for bias in [0.1, 0.3, 0.5, 0.7, 0.9]:
        chain = _grow_greedy(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    # Random walk growth
    for _ in range(5):
        chain = _grow_random_walk(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    # If no valid chains, use a straight line
    if not initial_chains:
        initial_chains = [[tuple([i] + [0] * (dimension - 1)) for i in range(n)]]
    
    # Local search on each initial conformation
    time_per_chain = per_sequence_time / max(len(initial_chains), 1)
    
    for chain in initial_chains:
        if time.time() - global_start > total_time_limit - 5.0:
            break
        
        occupied = {p: i for i, p in enumerate(chain)}
        chain, energy = _local_search(sequence, chain, occupied, time_per_chain, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Final fallback
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
