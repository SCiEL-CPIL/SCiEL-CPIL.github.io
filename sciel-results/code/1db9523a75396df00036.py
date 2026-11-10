"""HP lattice protein folding with local search and time-aware budgeting."""
# EVOLVE-BLOCK-START
import time
import random
import math

_TOTAL_DEADLINE = 180.0  # seconds, safety margin below 210s kill
_START_TIME = time.time()
_TIME_MARGIN = 10.0  # seconds to reserve


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _TOTAL_DEADLINE - _TIME_MARGIN - _elapsed())


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: minus the number of H-H contacts (non-adjacent along chain)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # each contact counted twice


def _is_valid_walk(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for p in chain:
        if p in seen:
            return False
        seen.add(p)
    for i in range(n - 1):
        dx = abs(chain[i][0] - chain[i+1][0])
        dy = abs(chain[i][1] - chain[i+1][1])
        if dx + dy != 1:
            return False
    return True


def _initial_spiral(sequence, dimension, rng):
    """Generate a compact spiral-like initial conformation."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Try to create a compact shape
    # Start with a simple pattern and randomize
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # Spiral pattern
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dir_idx = 0
    steps_in_dir = 0
    max_steps = 2
    
    for i in range(1, n):
        placed = False
        for attempt in range(20):
            d = directions[dir_idx % 4]
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                steps_in_dir += 1
                placed = True
                break
            # Try next direction
            dir_idx += 1
            steps_in_dir = 0
            max_steps = min(max_steps + 1, 6)
        
        if not placed:
            # Fallback: try all neighbors
            for d in directions:
                new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied.add(new_pos)
                    dir_idx = directions.index(d)
                    steps_in_dir = 1
                    break
            else:
                # Cannot extend, just stop (shouldn't happen for reasonable n)
                break
        
        # Change direction based on spiral logic
        if steps_in_dir >= max_steps:
            dir_idx += 1
            steps_in_dir = 0
            max_steps = min(max_steps + 1, 8)
    
    # Pad if needed
    while len(chain) < n:
        found = False
        for d in directions:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied.add(new_pos)
                found = True
                break
        if not found:
            break
    
    return chain[:n]


def _local_search(sequence, chain, rng, time_limit):
    """Improve chain using local search moves."""
    n = len(chain)
    if n <= 2:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        move_type = rng.random()
        
        if move_type < 0.3:
            # Pivot move: rotate a segment
            if n < 3:
                continue
            i = rng.randint(0, n - 3)
            j = rng.randint(i + 1, n - 1)
            segment = chain[i+1:j+1]
            
            # Try rotating segment around chain[i]
            orig_i = chain[i]
            new_segment = []
            valid = True
            for k, p in enumerate(segment):
                # Rotate 90 degrees around orig_i
                dx = p[0] - orig_i[0]
                dy = p[1] - orig_i[1]
                # 90 degree rotation: (dx, dy) -> (-dy, dx)
                new_p = (orig_i[0] - dy, orig_i[1] + dx)
                new_segment.append(new_p)
            
            # Check validity
            test_chain = chain[:i+1] + new_segment + chain[j+1:]
            if _is_valid_walk(test_chain):
                new_energy = _energy(sequence, test_chain)
                if new_energy <= best_energy or rng.random() < math.exp((new_energy - current_energy) / temperature):
                    chain = test_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
        
        elif move_type < 0.6:
            # Segment reversal
            if n < 3:
                continue
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            segment = chain[i:j+1]
            # Reverse segment but keep first and last connected
            new_segment = [segment[0]] + segment[2:][::-1] if len(segment) > 2 else segment[::-1]
            # Actually, just reverse the middle
            if j - i >= 2:
                new_segment = [segment[0]] + segment[1:][::-1]
            else:
                new_segment = segment[::-1]
            
            test_chain = chain[:i] + new_segment + chain[j+1:]
            if _is_valid_walk(test_chain):
                new_energy = _energy(sequence, test_chain)
                if new_energy <= best_energy or rng.random() < math.exp((new_energy - current_energy) / temperature):
                    chain = test_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
        
        else:
            # Single residue shift (move one residue to adjacent free site)
            if n < 3:
                continue
            i = rng.randint(1, n - 2)  # Don't move ends
            p = chain[i]
            neighbors = list(_neighbours(p))
            rng.shuffle(neighbors)
            for new_p in neighbors:
                # Check if new_p is free (not occupied by other residues)
                occupied_set = set(chain)
                occupied_set.remove(p)
                if new_p in occupied_set:
                    continue
                # Check connectivity: chain[i-1] must connect to new_p
                prev_p = chain[i-1]
                next_p = chain[i+1]
                if abs(prev_p[0] - new_p[0]) + abs(prev_p[1] - new_p[1]) != 1:
                    continue
                if abs(next_p[0] - new_p[0]) + abs(next_p[1] - new_p[1]) != 1:
                    continue
                
                test_chain = list(chain)
                test_chain[i] = new_p
                if _is_valid_walk(test_chain):
                    new_energy = _energy(sequence, test_chain)
                    if new_energy <= best_energy or rng.random() < math.exp((new_energy - current_energy) / temperature):
                        chain = test_chain
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_chain = list(chain)
                        break
        
        temperature = max(0.01, temperature * cooling)
    
    return best_chain


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
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


def fold(sequence, dimension):
    """Best conformation using time-adaptive search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Estimate time per sequence based on remaining time
    # Assume up to 10 sequences, allocate proportionally
    time_per_seq = _remaining() / 10.0
    time_per_seq = min(time_per_seq, 15.0)  # Cap at 15s per sequence
    
    # Generate initial candidates
    candidates = []
    
    # 1. Spiral initialization
    spiral = _initial_spiral(sequence, dimension, rng)
    if len(spiral) == n and _is_valid_walk(spiral):
        candidates.append(spiral)
    
    # 2. Multiple greedy growths
    num_growths = max(5, int(time_per_seq * 2))  # Scale with time
    for _ in range(num_growths):
        chain = _grow(sequence, dimension, rng)
        if chain is not None and _is_valid_walk(chain):
            candidates.append(chain)
    
    if not candidates:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Evaluate all candidates
    best_chain = None
    best_energy = float('inf')
    for cand in candidates:
        e = _energy(sequence, cand)
        if e < best_energy:
            best_energy = e
            best_chain = cand
    
    # Local search on best candidate
    if best_chain is not None and _remaining() > 1.0:
        # Allocate remaining time for local search
        ls_time = min(_remaining() * 0.5, 5.0)
        improved = _local_search(sequence, best_chain, rng, ls_time)
        e = _energy(sequence, improved)
        if e < best_energy:
            best_energy = e
            best_chain = improved
    
    if best_chain is None:
        return [[i, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
