"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


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


def _delta_energy_local(sequence, chain, where, idx, new_site):
    """Compute energy delta for moving residue idx from old_site to new_site."""
    old_site = chain[idx]
    if old_site == new_site:
        return 0
    
    delta = 0
    is_h = sequence[idx] == 'H'
    
    if is_h:
        # Remove contacts with neighbors of old_site
        for q in _neighbours(old_site):
            j = where.get(q)
            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                delta += 1
        
        # Add contacts with neighbors of new_site
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                delta -= 1
    
    return delta


def _grow(sequence, dimension, rng):
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


def _local_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a single-residue move. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    # Candidate new sites: neighbours of old_site that are adjacent to both prev and next
    candidates = []
    for site in _neighbours(old_site):
        if site == old_site:
            continue
        # Must be adjacent to prev and next
        if abs(site[0] - prev_site[0]) + abs(site[1] - prev_site[1]) != 1:
            continue
        if abs(site[0] - next_site[0]) + abs(site[1] - next_site[1]) != 1:
            continue
        # Self-avoidance check
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)

    # Perform move
    del where[old_site]
    where[new_site] = idx
    chain[idx] = new_site

    delta_e = _delta_energy_local(sequence, chain, where, idx, new_site)
    new_energy = current_energy + delta_e

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        del where[new_site]
        where[old_site] = idx
        chain[idx] = old_site
        return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # Check adjacency after reversal
    if i > 0:
        if abs(chain[i - 1][0] - chain[j][0]) + abs(chain[i - 1][1] - chain[j][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(chain[i][0] - chain[j + 1][0]) + abs(chain[i][1] - chain[j + 1][1]) != 1:
            return chain, where, current_energy, False

    # Perform reversal
    chain[i:j + 1] = chain[i:j + 1][::-1]
    for k in range(i, j + 1):
        where[chain[k]] = k

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        chain[i:j + 1] = chain[i:j + 1][::-1]
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _rotate_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a rotation move: pick a segment and rotate it by 90 degrees around one end.
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)
    
    # Try rotating the segment chain[i:j+1] around chain[i]
    # The segment endpoints are chain[i] and chain[j]
    # After rotation, chain[j] must be adjacent to chain[i+1] (if i > 0) and chain[i] must be adjacent to chain[j+1] (if j < n-1)
    
    # For simplicity, let's just do a 90-degree rotation of the segment around chain[i]
    # This means we need to transform all points in the segment
    cx, cy = chain[i]
    
    # Determine rotation direction (clockwise or counter-clockwise)
    direction = rng.choice([1, -1])
    
    def rotate_90(px, py, cx, cy, direction):
        dx = px - cx
        dy = py - cy
        if direction == 1:  # clockwise
            return (cx + dy, cy - dx)
        else:  # counter-clockwise
            return (cx - dy, cy + dx)
    
    # Check if the rotated segment would be valid
    new_segment = []
    for k in range(i, j + 1):
        nx, ny = rotate_90(chain[k][0], chain[k][1], cx, cy, direction)
        new_segment.append((nx, ny))
    
    # Check self-avoidance and adjacency
    # The new segment endpoints must connect properly
    if i > 0:
        if abs(new_segment[0][0] - chain[i-1][0]) + abs(new_segment[0][1] - chain[i-1][1]) != 1:
            return chain, where, current_energy, False
    if j < n - 1:
        if abs(new_segment[-1][0] - chain[j+1][0]) + abs(new_segment[-1][1] - chain[j+1][1]) != 1:
            return chain, where, current_energy, False
    
    # Check self-avoidance: no point in new_segment should be occupied by indices outside [i, j]
    for k in range(i, j + 1):
        site = new_segment[k - i]
        if site in where:
            occ_idx = where[site]
            if occ_idx < i or occ_idx > j:
                return chain, where, current_energy, False
    
    # Perform the move
    old_segment = chain[i:j+1]
    for k in range(i, j + 1):
        del where[chain[k]]
    chain[i:j+1] = new_segment
    for k in range(i, j + 1):
        where[chain[k]] = k
    
    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        for k in range(i, j + 1):
            del where[chain[k]]
        chain[i:j+1] = old_segment
        for k in range(i, j + 1):
            where[chain[k]] = k
        return chain, where, current_energy, False


def _anneal(sequence, dimension, chain, max_time=10.0):
    """Simulated annealing with local moves, reversals, and rotations."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    rng = random.Random(12345 + n + int(time.time()))

    start = time.time()
    while T > T_min and (time.time() - start) < max_time:
        # Alternate between local moves, reversals, and rotations
        r = rng.random()
        if r < 0.5:
            chain, where, current_energy, accepted = _local_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.8:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _rotate_move(sequence, chain, where, current_energy, T, rng)

        if accepted and current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)

        T *= cooling

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths followed by annealing."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1

    # Phase 1: Greedy growth to get initial conformations
    initial_chains = []
    for _ in range(50):
        if _remaining_time() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        return [list(p) for p in best]

    # Phase 2: Anneal each initial conformation
    time_per_anneal = max(1.0, _remaining_time() / len(initial_chains) * 0.8)
    for chain in initial_chains:
        if _remaining_time() < 5:
            break
        annealed_chain, annealed_energy = _anneal(sequence, dimension, chain, max_time=time_per_anneal)
        if annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = initial_chains[0]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
