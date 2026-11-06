"""HP lattice protein folding on the square lattice using hybrid local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

START_TIME = time.time()
TIME_LIMIT = 205.0  # Safety margin for 210s wall-clock limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _neighbours(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _energy(sequence, chain):
    """Calculate energy = -(number of non-consecutive H-H contacts)."""
    n = len(chain)
    if n == 0:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                j = pos_to_idx.get((x+dx, y+dy))
                if j is not None and j > i and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _is_valid(chain):
    """Check self-avoidance and unit steps."""
    if len(chain) != len(set(chain)):
        return False
    for i in range(len(chain) - 1):
        x1, y1 = chain[i]
        x2, y2 = chain[i+1]
        if abs(x1-x2) + abs(y1-y2) != 1:
            return False
    return True

def _grow(sequence, rng, bias_strength=3.0):
    """Randomized greedy growth with H-H clustering bias."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    h_indices = {i for i, res in enumerate(sequence) if res == 'H'}
    n = len(sequence)
    
    for i in range(1, n):
        last_pos = chain[-1]
        options = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            score = 0.0
            # Bonus for potential H-H contacts
            if i in h_indices:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j in h_indices and abs(i - j) > 1:
                            score += bias_strength
            # Prefer sites with more free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.3 * free
            score += rng.random() * 1.5
            options.append((score, site))
        
        if not options:
            return None
        
        best_site = max(options, key=lambda x: x[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain

def _pivot_move(chain, rng):
    """Try a pivot move: rotate a segment around a pivot point by 90 or 180 degrees."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose pivot index (not at ends)
    pivot = rng.randint(1, n - 2)
    
    # Choose segment length to rotate (from pivot+1 onward, but not too long)
    max_len = min(n - pivot - 1, rng.randint(2, min(10, n - pivot - 1)))
    seg_len = rng.randint(1, max_len)
    
    # Extract the segment to rotate: chain[pivot+1 : pivot+1+seg_len]
    seg_start = pivot + 1
    seg_end = seg_start + seg_len
    
    if seg_end >= n:
        return None
    
    segment = chain[seg_start:seg_end]
    pivot_pos = chain[pivot]
    
    # Try rotations: 90 CW, 90 CCW, 180
    rotations = [
        [(x+1, y) for x, y in segment],      # 90 CW
        [(x-1, y) for x, y in segment],      # 90 CCW
        [(x, y+1) for x, y in segment],      # 90 CW (other axis)
        [(x, y-1) for x, y in segment],      # 90 CCW (other axis)
        [(x+1, y+1) for x, y in segment],    # diagonal shift
        [(x-1, y-1) for x, y in segment],
        [(x+1, y-1) for x, y in segment],
        [(x-1, y+1) for x, y in segment],
    ]
    
    # Also try 180 degree rotation around pivot
    rot180 = [(2*pivot_pos[0]-x, 2*pivot_pos[1]-y) for x, y in segment]
    rotations.append(rot180)
    
    # Build set of occupied positions excluding the segment
    occupied_set = set(chain[:seg_start]) | set(chain[seg_end:])
    
    for rotated in rotations:
        # Check if rotated segment is valid
        valid = True
        for idx, pos in enumerate(rotated):
            if pos in occupied_set:
                valid = False
                break
            if idx > 0:
                px, py = rotated[idx-1]
                if abs(pos[0]-px) + abs(pos[1]-py) != 1:
                    valid = False
                    break
        if not valid:
            continue
        
        # Check connection at boundaries
        if seg_start > 0:
            prev_pos = chain[seg_start - 1]
            first_rot = rotated[0]
            if abs(first_rot[0]-prev_pos[0]) + abs(first_rot[1]-prev_pos[1]) != 1:
                continue
        if seg_end < n:
            next_pos = chain[seg_end]
            last_rot = rotated[-1]
            if abs(last_rot[0]-next_pos[0]) + abs(last_rot[1]-next_pos[1]) != 1:
                continue
        
        new_chain = chain[:seg_start] + rotated + chain[seg_end:]
        if _is_valid(new_chain):
            return new_chain
    
    return None

def _slither_move(chain, rng):
    """Try a slither move: shift a small segment perpendicular to its direction."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose a small segment of length 2-3
    seg_len = rng.randint(2, min(3, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    if end >= n:
        return None
    
    segment = chain[start:end]
    
    # Determine the direction of the segment
    dx = segment[-1][0] - segment[0][0]
    dy = segment[-1][1] - segment[0][1]
    
    # Perpendicular directions
    if dx != 0:
        perps = [(0, 1), (0, -1)]
    else:
        perps = [(1, 0), (-1, 0)]
    
    occupied_set = set(chain[:start]) | set(chain[end:])
    
    for pdx, pdy in perps:
        shifted = [(x+pdx, y+pdy) for x, y in segment]
        
        # Check validity
        valid = True
        for pos in shifted:
            if pos in occupied_set:
                valid = False
                break
        if not valid:
            continue
        
        # Check connections
        if start > 0:
            prev = chain[start-1]
            if abs(shifted[0][0]-prev[0]) + abs(shifted[0][1]-prev[1]) != 1:
                continue
        if end < n:
            nxt = chain[end]
            if abs(shifted[-1][0]-nxt[0]) + abs(shifted[-1][1]-nxt[1]) != 1:
                continue
        
        new_chain = chain[:start] + shifted + chain[end:]
        if _is_valid(new_chain):
            return new_chain
    
    return None

def _reversal_move(chain, rng):
    """Try a reversal move: reverse a segment of the chain."""
    n = len(chain)
    if n < 5:
        return None
    
    # Choose segment length
    seg_len = rng.randint(2, min(8, n - 2))
    start = rng.randint(0, n - seg_len)
    end = start + seg_len
    
    if end >= n:
        return None
    
    segment = chain[start:end]
    reversed_segment = segment[::-1]
    
    new_chain = chain[:start] + reversed_segment + chain[end:]
    if _is_valid(new_chain):
        return new_chain
    
    return None

def _local_search(sequence, chain, rng, time_budget):
    """Local search with multiple move types and simulated annealing acceptance."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    
    deadline = time.time() + time_budget
    temp = 3.0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and temp > 0.05:
        # Choose move type
        r = rng.random()
        new_chain = None
        
        if r < 0.4:
            new_chain = _pivot_move(current, rng)
        elif r < 0.7:
            new_chain = _slither_move(current, rng)
        else:
            new_chain = _reversal_move(current, rng)
        
        if new_chain is None:
            temp *= 0.998
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with Boltzmann probability
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = list(current)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Cool down
        temp *= 0.999
        
        # If stuck for too long, restart with a small perturbation
        if no_improve_count > max_no_improve:
            # Perturb: do a few random reversals
            for _ in range(3):
                perturbed = _reversal_move(current, rng)
                if perturbed:
                    current = perturbed
                    current_energy = _energy(sequence, current)
            no_improve_count = 0
            temp = max(temp, 1.0)
    
    return best

def _polish(sequence, chain):
    """Deterministic polish: try all simple moves and accept any energy improvement."""
    current = list(chain)
    current_energy = _energy(sequence, current)
    improved = True
    
    while improved:
        improved = False
        n = len(current)
        
        # Try all short reversals
        for seg_len in range(2, min(6, n - 1)):
            for start in range(0, n - seg_len):
                end = start + seg_len
                segment = current[start:end]
                rev_segment = segment[::-1]
                new_chain = current[:start] + rev_segment + current[end:]
                if _is_valid(new_chain):
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < current_energy:
                        current = new_chain
                        current_energy = new_energy
                        improved = True
                        break
            if improved:
                break
        
        # Try all slither moves on segments of length 2
        if not improved:
            for start in range(1, n - 2):
                segment = current[start:start+2]
                dx = segment[1][0] - segment[0][0]
                dy = segment[1][1] - segment[0][1]
                if dx != 0:
                    perps = [(0, 1), (0, -1)]
                else:
                    perps = [(1, 0), (-1, 0)]
                for pdx, pdy in perps:
                    shifted = [(x+pdx, y+pdy) for x, y in segment]
                    occupied_set = set(current[:start]) | set(current[start+2:])
                    if any(p in occupied_set for p in shifted):
                        continue
                    if start > 0:
                        prev = current[start-1]
                        if abs(shifted[0][0]-prev[0]) + abs(shifted[0][1]-prev[1]) != 1:
                            continue
                    if start+2 < n:
                        nxt = current[start+2]
                        if abs(shifted[1][0]-nxt[0]) + abs(shifted[1][1]-nxt[1]) != 1:
                            continue
                    new_chain = current[:start] + shifted + current[start+2:]
                    if _is_valid(new_chain):
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < current_energy:
                            current = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                if improved:
                    break
    
    return current

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    if _time_left() < 2.0:
        return [[i, 0] for i in range(len(sequence))]
    
    n = len(sequence)
    rng = random.Random(hash(sequence) % (2**32))
    
    best_chain = None
    best_energy = float('inf')
    
    # Generate multiple diverse initial conformations
    num_initial = max(5, min(30, 200 // n))
    initial_chains = []
    
    for attempt in range(num_initial):
        if _time_left() < 5.0:
            break
        bias = 2.0 + rng.random() * 3.0
        c = _grow(sequence, rng, bias_strength=bias)
        if c is not None:
            initial_chains.append(c)
    
    if not initial_chains:
        return [[i, 0] for i in range(len(sequence))]
    
    # Sort by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    # Allocate remaining time across local search passes
    remaining_time = _time_left() - 3.0  # reserve 3s for polish and return
    if remaining_time < 1.0:
        return [list(p) for p in initial_chains[0]]
    
    # Optimize top chains
    num_to_optimize = min(3, len(initial_chains))
    time_per_chain = remaining_time / num_to_optimize
    
    for i in range(num_to_optimize):
        if _time_left() < 5.0:
            break
        
        chain = initial_chains[i]
        budget = min(time_per_chain, _time_left() - 4.0)
        if budget < 0.5:
            break
        
        optimized = _local_search(sequence, chain, rng, budget)
        if optimized:
            e = _energy(sequence, optimized)
            if e < best_energy:
                best_energy = e
                best_chain = optimized
    
    # Final polish on the best chain
    if best_chain is None:
        best_chain = initial_chains[0]
    
    if _time_left() > 2.0:
        best_chain = _polish(sequence, best_chain)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
