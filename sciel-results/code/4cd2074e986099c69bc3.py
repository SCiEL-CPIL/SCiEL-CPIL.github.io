"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 175.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _calculate_energy(sequence, chain):
    """Compute the HP energy: -number of H-H contacts (non-consecutive, adjacent on lattice)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 3.0 + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, i, rng):
    """
    Try a pivot move at position i: rotate the tail (positions i..n-1) by 90 or 180 degrees
    around position i-1. Returns new chain if valid, else None.
    """
    n = len(chain)
    if i < 1 or i >= n:
        return None
    
    # Pick a random pivot angle: 90 CW, 90 CCW, or 180
    angle = rng.choice([1, -1, 2])  # 1=90CW, -1=90CCW, 2=180
    
    # The tail starts at position i
    tail = chain[i:]
    pivot = chain[i - 1]
    
    # Rotate each point in the tail around the pivot
    new_tail = []
    for p in tail:
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        if angle == 1:  # 90 CW: (dx,dy) -> (dy, -dx)
            ndx, ndy = dy, -dx
        elif angle == -1:  # 90 CCW: (dx,dy) -> (-dy, dx)
            ndx, ndy = -dy, dx
        else:  # 180: (dx,dy) -> (-dx, -dy)
            ndx, ndy = -dx, -dy
        new_tail.append((pivot[0] + ndx, pivot[1] + ndy))
    
    # Check self-avoidance
    new_chain = chain[:i] + new_tail
    # The head part (positions 0..i-1) is unchanged, so we only need to check:
    # 1. No two points in new_tail are the same
    # 2. No point in new_tail collides with a point in head (positions 0..i-1)
    # 3. Consecutive points in the full chain are adjacent (unit steps)
    
    # Check unit step between chain[i-1] and new_chain[i]
    if abs(new_chain[i][0] - chain[i-1][0]) + abs(new_chain[i][1] - chain[i-1][1]) != 1:
        return None
    
    # Check self-avoidance: build set of head positions
    head_set = set(chain[:i])
    seen = set()
    for j in range(i, n):
        p = new_chain[j]
        if p in head_set or p in seen:
            return None
        seen.add(p)
        # Check unit step between consecutive tail points
        if j > i:
            if abs(p[0] - new_chain[j-1][0]) + abs(p[1] - new_chain[j-1][1]) != 1:
                return None
    
    return new_chain


def _local_search(sequence, chain, rng, max_iter=2000):
    """
    Local search using pivot moves. Returns the best chain found.
    """
    n = len(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Pick a random pivot position (avoid too close to ends)
        min_i = 1
        max_i = n - 1
        i = rng.randint(min_i, max_i)
        
        new_chain = _pivot_move(chain, i, rng)
        if new_chain is None:
            continue
        
        new_energy = _calculate_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain


def _local_search2(sequence, chain, rng, max_iter=3000):
    """
    More aggressive local search with multiple pivot positions per step.
    """
    n = len(chain)
    current_energy = _calculate_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 0.5
    T_min = 0.05
    cooling = 0.998
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Try several random pivots
        found_improvement = False
        for _ in range(3):
            i = rng.randint(1, n - 1)
            new_chain = _pivot_move(chain, i, rng)
            if new_chain is None:
                continue
            
            new_energy = _calculate_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy = new_energy
                found_improvement = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                break
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation found using multiple strategies:
    1. Multiple randomized greedy growths
    2. Local search with pivot moves on the best results
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Strategy 1: Multiple greedy growths
    best_chain = None
    best_energy = 0  # Start at 0 (unfolded)
    
    # Number of growths depends on time budget and sequence length
    num_growths = min(500, max(50, 20000 // n))
    
    for _ in range(num_growths):
        if _time_left() < 10.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _calculate_energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Strategy 2: Local search on the best chain
    if _time_left() > 20.0:
        # Do local search with different random seeds
        for seed_offset in range(5):
            if _time_left() < 10.0:
                break
            rng_ls = random.Random(99999 + len(sequence) + seed_offset)
            improved = _local_search(sequence, best_chain, rng_ls, max_iter=1500)
            improved_energy = _calculate_energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
    
    # Strategy 3: More aggressive local search if time remains
    if _time_left() > 15.0:
        rng_ls2 = random.Random(88888 + len(sequence))
        improved = _local_search2(sequence, best_chain, rng_ls2, max_iter=2000)
        improved_energy = _calculate_energy(sequence, improved)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved
    
    # Strategy 4: Additional greedy + local search cycles if time permits
    while _time_left() > 25.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _calculate_energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = list(chain)
        
        # Quick local search
        if _time_left() > 15.0:
            rng_ls3 = random.Random(77777 + len(sequence) + rng.random())
            improved = _local_search(sequence, best_chain, rng_ls3, max_iter=500)
            improved_energy = _calculate_energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
