"""HP lattice protein folding on the cubic lattice.
Strategy: Multi-Start Simulated Annealing (SA) over Pivot Moves.
"""
# EVOLVE-BLOCK-START
import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _random_fold(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = [site for site in _neighbours(last) if site not in occupied]
        if not options:
            return None
        site = rng.choice(options)
        chain.append(site)
        occupied[site] = i
    return chain

def _pivot_move(chain, rng):
    """Apply a pivot move to the chain. Returns (new_chain, valid)."""
    n = len(chain)
    if n < 3:
        return chain, True
    
    # Pick a pivot point
    pivot_idx = rng.randint(1, n - 1)
    
    # Pick a random axis and direction
    axis = rng.randint(0, len(chain[0]) - 1)
    direction = rng.choice([-1, 1])
    
    # Extract the tail
    tail = chain[pivot_idx:]
    head = chain[:pivot_idx]
    
    # Rotate the tail
    new_tail = []
    for p in tail:
        new_p = list(p)
        new_p[axis] += direction
        new_tail.append(tuple(new_p))
    
    new_chain = head + new_tail
    
    # Check for self-avoidance
    # We only need to check the new tail against the rest of the chain (head + rest of tail)
    # Actually, we can just check the whole new chain for duplicates, but that's O(N^2).
    # More efficient: check new_tail against (head + new_tail[:-1])? No, new_tail is rotated as a rigid body.
    # The internal structure of new_tail is preserved. So we only need to check new_tail against head.
    # And check new_tail against itself? No, rotation preserves distances within the tail.
    # So we just need to ensure no point in new_tail is in head.
    
    occupied_head = set(head)
    for p in new_tail:
        if p in occupied_head:
            return chain, False
    
    return new_chain, True

def _energy(sequence, chain):
    """Calculate the energy of the chain (negative number of H-H contacts)."""
    if not chain:
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

def _sa_fold(sequence, dimension, rng, time_limit):
    """Run Simulated Annealing for a given time limit."""
    start_time = time.time()
    
    # Initial random fold
    chain = _random_fold(sequence, dimension, rng)
    if chain is None:
        return None, 1 # Return a straight line as fallback? Or None.
        
    # Fallback to straight line if random fold fails
    if chain is None:
        chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        new_chain, valid = _pivot_move(chain, rng)
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation found via Multi-Start SA."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Allocate time based on sequence length
    # Longer chains need more time per residue, but we have a global budget.
    # Let's allocate time proportional to length, with a minimum.
    # Total budget for all sequences is ~330s.
    # We don't know how many sequences will be called, but typically ~26.
    # Let's assume we can spend ~10s per sequence on average.
    # For short chains, less time is needed.
    time_budget = 0.5 + 0.05 * n
    if time_budget > 10.0:
        time_budget = 10.0
        
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so 0 is worse than any valid fold
    
    # Multi-start: try a few different SA runs
    # If n is small, we can do more runs.
    num_runs = max(1, int(2.0 / (time_budget / 10.0))) # Scale runs inversely with time per run?
    # Actually, just do 1 run with the allocated time, or split time among 2-3 runs.
    # Let's do 2 runs, splitting the time.
    
    runs = 2 if time_budget > 1.0 else 1
    time_per_run = time_budget / runs
    
    for _ in range(runs):
        chain, energy = _sa_fold(sequence, dimension, rng, time_per_run)
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
