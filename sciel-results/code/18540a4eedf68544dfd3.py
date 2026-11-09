"""HP lattice protein folding - optimized with simulated annealing and multi-start."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 200.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, i, old_pos, new_pos):
    """Compute energy delta when moving position i from old_pos to new_pos.
    This is an approximation - for correctness we recompute full energy."""
    # For simplicity and correctness, just recompute
    new_chain = chain.copy()
    new_chain[i] = new_pos
    return _energy(sequence, new_chain) - _energy(sequence, chain)


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick with some randomness - sometimes take best, sometimes random
        if rng.random() < 0.7:
            site = max(options)[1]
        else:
            site = options[rng.randint(0, len(options) - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_move(chain, rng):
    """Perform a local move. Returns new chain or None.
    Uses pivot moves (rotate tail segment) for better exploration."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot point
    pivot = rng.randint(1, n - 3)
    
    # Get the tail segment
    tail_start = chain[pivot + 1]
    dx = tail_start[0] - chain[pivot][0]
    dy = tail_start[1] - chain[pivot][1]
    
    # The 3 possible new directions (90, 180, 270 degree rotations)
    # Current direction is (dx, dy)
    rotations = [
        (-dy, dx),    # 90 degrees CCW
        (-dx, -dy),   # 180 degrees
        (dy, -dx),    # 270 degrees CCW (or 90 CW)
    ]
    
    # Try each rotation
    for new_dx, new_dy in rotations:
        # Check if this is the same as current direction
        if new_dx == dx and new_dy == dy:
            continue
        
        # Build new tail
        new_tail = [(chain[pivot][0] + new_dx, chain[pivot][1] + new_dy)]
        
        # Continue the tail maintaining relative positions
        # Actually, we need to rotate the entire tail around the pivot
        # Let's do it properly: rotate each point in the tail
        
        # The tail is chain[pivot+1:]
        # We rotate the direction of the first step, then continue
        # But this is complex. Let's use a simpler approach:
        # Try moving a single residue
        
        break  # Skip complex pivot for now, use simpler moves
    
    # Simpler: try to move a random internal residue
    i = rng.randint(1, n - 2)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(directions)
    
    for dx, dy in directions:
        new_pos = (chain[i][0] + dx, chain[i][1] + dy)
        
        # Check unit step to previous
        if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
            continue
        # Check unit step to next
        if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
            continue
        
        # Check no collision
        collides = False
        for j, pos in enumerate(chain):
            if j == i:
                continue
            if pos == new_pos:
                collides = True
                break
        
        if not collides:
            new_chain = chain.copy()
            new_chain[i] = new_pos
            return new_chain
    
    return None


def _pivot_move(chain, rng):
    """Pivot move: rotate the tail segment around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    pivot = rng.randint(1, n - 3)
    
    # The segment from pivot+1 to end
    # Get the direction from pivot to pivot+1
    dx = chain[pivot + 1][0] - chain[pivot][0]
    dy = chain[pivot + 1][1] - chain[pivot][1]
    
    # Possible new directions (excluding current)
    all_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    new_dirs = [d for d in all_dirs if d != (dx, dy)]
    
    if not new_dirs:
        return None
    
    new_dx, new_dy = rng.choice(new_dirs)
    
    # Build new tail by rotating the entire tail
    # The tail is chain[pivot+1:]
    # We need to rotate each point relative to the pivot
    
    # Actually, a proper pivot move rotates the entire tail rigidly
    # Let's compute the new positions
    
    # The tail points relative to chain[pivot]:
    # tail[0] is at (dx, dy) from pivot
    # tail[1] is at (dx + (chain[pivot+2][0]-chain[pivot+1][0]), dy + (chain[pivot+2][1]-chain[pivot+1][1]))
    # etc.
    
    # To rotate, we rotate the vector from pivot to each tail point
    # Rotation by 90 deg CCW: (x, y) -> (-y, x)
    # Rotation by 180 deg: (x, y) -> (-x, -y)
    # Rotation by 270 deg CCW: (x, y) -> (y, -x)
    
    # Determine which rotation
    if (new_dx, new_dy) == (-dy, dx):
        rot = 1  # 90 CCW
    elif (new_dx, new_dy) == (-dx, -dy):
        rot = 2  # 180
    else:
        rot = 3  # 270 CCW
    
    new_tail = []
    for k in range(pivot + 1, n):
        rx = chain[k][0] - chain[pivot][0]
        ry = chain[k][1] - chain[pivot][1]
        
        if rot == 1:
            nrx, nry = -ry, rx
        elif rot == 2:
            nrx, nry = -rx, -ry
        else:
            nrx, nry = ry, -rx
        
        new_tail.append((chain[pivot][0] + nrx, chain[pivot][1] + nry))
    
    # Check for collisions with head (chain[:pivot+1])
    head_set = set(chain[:pivot + 1])
    valid = True
    for pos in new_tail:
        if pos in head_set:
            valid = False
            break
    
    if not valid:
        return None
    
    new_chain = chain[:pivot + 1] + new_tail
    return new_chain


def _anneal(sequence, chain, rng, time_limit):
    """Simulated annealing with time-based termination."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    T_start = 3.0
    T_end = 0.05
    start_time = time.time()
    
    step = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit and _remaining_time() > 1.0:
        step += 1
        
        # Temperature schedule
        progress = min(1.0, (time.time() - start_time) / time_limit)
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move type
        if rng.random() < 0.5 and n >= 4:
            new_chain = _pivot_move(chain, rng)
        else:
            new_chain = _local_move(chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < exp(-delta / max(T, 0.01)):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
        
        # Restart if stuck too long
        if no_improve > max_no_improve:
            # Try a new growth
            new_growth = _grow(sequence, 2, rng)
            if new_growth is not None:
                ge = _energy(sequence, new_growth)
                if ge < current_energy or rng.random() < 0.3:
                    chain = new_growth
                    current_energy = ge
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                    no_improve = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    rng = random.Random(42 + len(sequence))
    
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    # Generate multiple initial chains
    initial_chains = []
    for _ in range(30):
        chain = _grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: simple zigzag
        chain = []
        for i in range(n):
            if i % 2 == 0:
                chain.append([i // 2, 0])
            else:
                chain.append([i // 2, 1])
        return chain
    
    # Find best initial
    best_chain = min(initial_chains, key=lambda c: _energy(sequence, c))
    best_energy = _energy(sequence, best_chain)
    
    # Allocate time for annealing runs
    time_per_run = max(2.0, (_remaining_time() - 2.0) / 8.0)
    
    # Run multiple annealing runs
    for run in range(8):
        if _remaining_time() < 1.5:
            break
        
        # Choose starting point
        if run == 0:
            start_chain = best_chain[:]
        elif rng.random() < 0.4:
            start_chain = _grow(sequence, dimension, rng)
            if start_chain is None:
                start_chain = best_chain[:]
        else:
            start_chain = best_chain[:]
        
        # Run annealing
        improved_chain, improved_energy = _anneal(sequence, start_chain, rng, time_per_run)
        
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
