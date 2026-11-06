"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 195s as a safe deadline to ensure we finish before kill.
    return 195.0 - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random() * 0.5, site))
        if not options:
            return None
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain


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


def _kink_move(chain, i, direction, occupied):
    """
    Try a kink move at position i.
    direction: 0=left turn, 1=right turn relative to current direction.
    Returns (new_chain, new_occupied) if valid, else (None, None).
    """
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None, None
    
    # Get the three consecutive points
    p0 = chain[i-1]
    p1 = chain[i]
    p2 = chain[i+1]
    
    # Current direction vector
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    # Rotate direction by 90 degrees
    if direction == 0:  # left turn
        new_dx, new_dy = -dy, dx
    else:  # right turn
        new_dx, new_dy = dy, -dx
    
    new_p1 = (p1[0] + new_dx, p1[1] + new_dy)
    
    # Check if new position is occupied (excluding the positions being modified)
    if new_p1 in occupied and new_p1 not in (p0, p1, p2):
        return None, None
    
    # Also check that new_p1 doesn't collide with any other chain position
    # (the occupied set check above handles this since we exclude p0, p1, p2)
    
    new_chain = chain[:]
    new_chain[i] = new_p1
    new_occupied = occupied.copy()
    del new_occupied[p1]
    new_occupied[new_p1] = i
    
    return new_chain, new_occupied


def _local_search(sequence, chain, max_iters=100000, rng=None):
    """
    Simulated annealing with kink moves.
    Returns the best chain found.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    for _ in range(max_iters):
        # Pick a random position for kink
        i = rng.randint(1, n - 2)
        direction = rng.randint(0, 1)
        
        new_chain, new_occupied = _kink_move(chain, i, direction, occupied)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > T_min and rng.random() < 2.718281828 ** (-delta / T)):
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        
        T *= cooling
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using greedy growth + local search, time-managed."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Phase 1: Multiple greedy growths to get diverse starting points
    deadline = _time_remaining()
    
    # Do at least 50 greedy growths, but time-managed
    num_growths = 0
    while _time_remaining() > 5.0 and num_growths < 100:
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            value = _energy(sequence, chain)
            if value < best_energy:
                best_energy = value
                best_chain = chain[:]
        num_growths += 1
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(len(sequence))]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search refinement on the best chain and some others
    # Keep a pool of starting chains for diversity
    pool = [best_chain[:]]
    
    # Add a few more diverse starting points
    while _time_remaining() > 15.0 and len(pool) < 5:
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            pool.append(chain)
    
    # Refine each chain in the pool with local search
    for idx, start_chain in enumerate(pool):
        if _time_remaining() < 3.0:
            break
        
        # Allocate remaining time evenly among remaining chains
        remaining_chains = len(pool) - idx
        time_per_chain = max(1.0, (_time_remaining() - 3.0) / remaining_chains)
        
        # Estimate iterations based on time
        # Each kink move + energy calc is ~microseconds, so we can do many
        # Be conservative: assume ~100k iters per second
        max_iters = int(time_per_chain * 50000)
        max_iters = max(1000, min(max_iters, 500000))
        
        refined = _local_search(sequence, start_chain, max_iters, rng)
        refined_energy = _energy(sequence, refined)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined
    
    # Final: if we still have time, do one more long local search on the best
    if _time_remaining() > 5.0 and best_chain is not None:
        max_iters = int(_time_remaining() * 80000)
        max_iters = max(10000, min(max_iters, 1000000))
        refined = _local_search(sequence, best_chain, max_iters, rng)
        refined_energy = _energy(sequence, refined)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
