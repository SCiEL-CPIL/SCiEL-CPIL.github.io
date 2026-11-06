"""Improved HP lattice folding with local search and time management."""
# EVOLVE-BLOCK-START
import random
import time

# Track global start time for deadline management
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Leave 15s safety margin before 210s wall-clock limit
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _local_search(sequence, chain, rng, max_iters=1000):
    """Local search: try moves that preserve self-avoidance and improve energy."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    where = {p: i for i, p in enumerate(current)}
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
        # Pick a random internal residue to move
        i = rng.randint(1, len(current) - 2)
        old_pos = current[i]
        # Try all 4 neighbors
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            # Check that moving residue i to new_pos doesn't create a collision
            # and that the new position is adjacent to both neighbors in chain
            # Actually, we need to ensure the chain remains a valid path
            # The move is valid if:
            # 1. new_pos is not occupied (checked)
            # 2. new_pos is adjacent to current[i-1] and current[i+1]
            # Wait, that's not right. We're moving a single residue, so the chain
            # edges change. We need new_pos to be adjacent to both i-1 and i+1.
            # Actually no: the chain is a path. If we move residue i, the edges
            # (i-1, i) and (i, i+1) change. So new_pos must be adjacent to
            # current[i-1] AND current[i+1].
            if i > 0 and not _is_adjacent(new_pos, current[i-1]):
                continue
            if i < len(current) - 1 and not _is_adjacent(new_pos, current[i+1]):
                continue
            
            # Tentatively move
            current[i] = new_pos
            del where[old_pos]
            where[new_pos] = i
            
            new_energy = _energy(sequence, current)
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                # Revert
                current[i] = old_pos
                del where[new_pos]
                where[old_pos] = i
    
    return current

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

def _pivot_search(sequence, chain, rng, max_iters=500):
    """Pivot-based local search: rotate a subchain."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
        # Pick a pivot point
        if len(current) < 4:
            break
        i = rng.randint(1, len(current) - 2)
        # Pick a subchain to rotate: from i to j
        j = rng.randint(i + 1, len(current) - 1)
        
        # Try rotating the subchain [i+1, j] around point i
        # We need to check if the rotated subchain fits without collision
        # This is complex; let's do a simpler move: try moving residue i+1
        # Actually, let's do a different local move: try swapping two non-adjacent
        # residues if it preserves the path property. That's hard.
        
        # Let's do a "kink" move: for a 3-residue segment, try the other kink
        if i < len(current) - 2:
            # Segment: current[i-1], current[i], current[i+1]
            # If they form a kink, try the other direction
            p0 = current[i-1]
            p1 = current[i]
            p2 = current[i+1]
            # The other kink position
            # If p0 and p2 are diagonal, the other kink is the remaining neighbor
            dx0 = p0[0] - p1[0]
            dy0 = p0[1] - p1[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            # The other position is (p1[0] + dx0 + dx2, p1[1] + dy0 + dy2)
            new_pos = (p1[0] + dx0 + dx2, p1[1] + dy0 + dy2)
            if new_pos != p1 and new_pos not in {current[k] for k in range(len(current)) if k != i}:
                # Check if new_pos is adjacent to both p0 and p2
                if _is_adjacent(new_pos, p0) and _is_adjacent(new_pos, p2):
                    current[i] = new_pos
                    new_energy = _energy(sequence, current)
                    if new_energy < current_energy:
                        current_energy = new_energy
                    else:
                        current[i] = p1
    
    return current

def fold(sequence, dimension):
    """Best conformation using greedy growth + local search with time management."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Determine number of restarts based on sequence length and time
    # For short sequences, do many restarts; for long, fewer
    if n <= 20:
        num_restarts = 200
    elif n <= 50:
        num_restarts = 100
    else:
        num_restarts = 50
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search
        chain = _local_search(sequence, chain, rng, max_iters=max(50, 500 - n * 2))
        chain = _pivot_search(sequence, chain, rng, max_iters=max(20, 200 - n))
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
