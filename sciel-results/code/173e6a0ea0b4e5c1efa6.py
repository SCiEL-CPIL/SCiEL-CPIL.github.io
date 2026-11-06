"""HP lattice protein folding on the square lattice using advanced local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 195.0  # Safety margin for 210s limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _neighbours(point):
    """Yield the 4 neighbors of a 2D lattice point."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Calculate the energy of a conformation.
    Energy = - (number of non-consecutive H-H contacts)
    """
    if not chain:
        return 0
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:
                        contacts += 1
    return -contacts

def _is_valid(chain):
    """Check if the chain is a valid self-avoiding walk."""
    if len(chain) != len(set(chain)):
        return False
    for i in range(len(chain) - 1):
        x1, y1 = chain[i]
        x2, y2 = chain[i + 1]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
    return True

def _grow_constrained(sequence, dimension, rng, bias=0.7):
    """
    Generate a random self-avoiding walk with a bias towards H-H contacts.
    bias: probability of choosing a move that increases H-H contacts (if available).
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    h_indices = {i for i, res in enumerate(sequence) if res == 'H'}
    
    for i in range(1, len(sequence)):
        last_pos = chain[-1]
        options = []
        
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Calculate score
            score = 0.0
            
            # 1. Strong bias for H-H contacts
            if i in h_indices:
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j in h_indices and abs(i - j) > 1:
                            score += 10.0
            
            # 2. Mild bias for free space (avoid dead ends)
            free_space = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.5 * free_space
            
            # 3. Random noise
            score += rng.random() * 3.0
            
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score descending and pick with bias
        options.sort(key=lambda x: x[0], reverse=True)
        
        # With probability 'bias', pick the best option. Otherwise, pick randomly.
        if rng.random() < bias:
            best_site = options[0][1]
        else:
            # Weighted random selection or just uniform? Uniform is safer for diversity.
            best_site = rng.choice(options)[1]
            
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain

def _pivot_move(chain, pivot_idx, rng):
    """
    Apply a pivot move: rotate the segment chain[pivot_idx+1:] by 90 or 180 degrees.
    Returns the new chain if valid, else None.
    """
    if pivot_idx <= 0 or pivot_idx >= len(chain) - 1:
        return None
        
    prefix = chain[:pivot_idx + 1]
    suffix = chain[pivot_idx + 1:]
    
    # The pivot point is prefix[-1]
    pivot_point = prefix[-1]
    
    # The first step of the suffix is from pivot_point to suffix[0]
    # dx, dy = suffix[0][0] - pivot_point[0], suffix[0][1] - pivot_point[1]
    # We rotate the entire suffix relative to the pivot point.
    
    # Rotation matrices for 2D:
    # 90 deg CW: (x, y) -> (y, -x)
    # 90 deg CCW: (x, y) -> (-y, x)
    # 180 deg: (x, y) -> (-x, -y)
    
    # We need to rotate the vector (dx, dy) of each point in suffix relative to pivot_point.
    # Actually, it's easier to rotate the coordinates relative to the pivot.
    
    px, py = pivot_point
    rotated_suffix = []
    
    # Choose a random rotation: 90, 180, or 270
    rot = rng.choice([1, 2, 3]) # 1=90, 2=180, 3=270
    
    for p in suffix:
        dx = p[0] - px
        dy = p[1] - py
        
        if rot == 1: # 90 deg
            new_dx = dy
            new_dy = -dx
        elif rot == 2: # 180 deg
            new_dx = -dx
            new_dy = -dy
        else: # 270 deg
            new_dx = -dy
            new_dy = dx
            
        rotated_suffix.append((px + new_dx, py + new_dy))
        
    new_chain = prefix + rotated_suffix
    
    # Check validity
    if not _is_valid(new_chain):
        return None
        
    return new_chain

def _segment_shift(chain, i, j, rng):
    """
    Shift a segment chain[i:j+1] by one step in a random direction.
    Returns the new chain if valid, else None.
    """
    if i < 0 or j >= len(chain) or i > j:
        return None
        
    segment = chain[i:j+1]
    
    # Determine possible shifts
    # The segment must move as a whole.
    # The connection at i-1 to i and j to j+1 must remain valid.
    
    # Try all 4 directions
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rng.shuffle(directions)
    
    for dx, dy in directions:
        shifted_segment = [(x + dx, y + dy) for x, y in segment]
        
        # Check boundaries
        if i > 0:
            # Check connection from i-1 to shifted i
            prev = chain[i-1]
            first = shifted_segment[0]
            if abs(prev[0] - first[0]) + abs(prev[1] - first[1]) != 1:
                continue
                
        if j < len(chain) - 1:
            # Check connection from shifted j to j+1
            next_ = chain[j+1]
            last = shifted_segment[-1]
            if abs(next_[0] - last[0]) + abs(next_[1] - last[1]) != 1:
                continue
            
        new_chain = chain[:i] + shifted_segment + chain[j+1:]
        
        if _is_valid(new_chain):
            return new_chain
            
    return None

def _slither_move(chain, rng):
    """
    Slither move: Move the end of the chain by one step if possible, 
    or perform a simple local move.
    Actually, a better 'local' move is to try to move a random internal bead 
    if it doesn't break connectivity (hard) or just stick to pivot/shift.
    Let's implement a 'Crankshaft' or just stick to Pivot and Shift which are robust.
    """
    return None

def _local_search(sequence, chain, time_budget, rng):
    """
    Perform local search using Pivot and Segment Shift moves.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    deadline = time.time() + time_budget
    n = len(current_chain)
    
    while time.time() < deadline:
        # Choose move type
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            # Pivot
            pivot_idx = rng.randint(1, n - 2)
            new_chain = _pivot_move(current_chain, pivot_idx, rng)
        elif move_type < 0.8:
            # Segment Shift
            # Choose a random segment length
            seg_len = rng.randint(1, min(10, n // 2))
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            new_chain = _segment_shift(current_chain, i, j, rng)
        else:
            # Random Pivot on short segment
            pivot_idx = rng.randint(1, min(5, n - 2))
            new_chain = _pivot_move(current_chain, pivot_idx, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Accept if better or with some probability (greedy hill climbing with occasional jumps)
        if delta_e < 0 or (delta_e == 0 and rng.random() < 0.5):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return best_chain

def _polish(sequence, chain, time_budget, rng):
    """
    Intensive local search with multiple restarts or just a longer run.
    """
    return _local_search(sequence, chain, time_budget, rng)

def _spiral_fold(sequence, rng):
    """
    Generate a spiral-like conformation.
    """
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    # Simple spiral pattern
    dx, dy = 1, 0
    step_len = 1
    
    for i in range(1, n):
        # Try to continue in current direction
        for _ in range(step_len):
            if i >= n:
                break
            nx, ny = chain[-1][0] + dx, chain[-1][1] + dy
            if (nx, ny) not in occupied:
                chain.append((nx, ny))
                occupied[(nx, ny)] = i
                i += 1
            else:
                # Turn
                break
                
        if i >= n:
            break
            
        # Turn left (90 degrees)
        dx, dy = -dy, dx
        step_len += 1
        
    # If we didn't place all residues, fill in randomly
    while len(chain) < n:
        last = chain[-1]
        options = [p for p in _neighbours(last) if p not in occupied]
        if not options:
            break
        next_p = rng.choice(options)
        chain.append(next_p)
        occupied[next_p] = len(chain) - 1
        
    if len(chain) != n:
        return None
        
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        return [[i, 0] for i in range(len(sequence))]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # 1. Generate diverse initial conformations
    initial_chains = []
    
    # Try constrained growth with different biases
    for bias in [0.5, 0.7, 0.9]:
        for _ in range(3):
            c = _grow_constrained(sequence, dimension, rng, bias)
            if c:
                initial_chains.append(c)
                
    # Try spiral fold
    c = _spiral_fold(sequence, rng)
    if c:
        initial_chains.append(c)
        
    # Random growth (no bias)
    for _ in range(2):
        c = _grow_constrained(sequence, dimension, rng, 0.0)
        if c:
            initial_chains.append(c)

    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(len(sequence))]
    
    # Sort initial chains by energy
    initial_chains.sort(key=lambda c: _energy(sequence, c))
    
    # 2. Optimize top candidates
    # We have ~195s total. 10 sequences. ~19s per sequence.
    # Let's allocate time dynamically.
    
    time_per_seq = TIME_LIMIT / 10.0 - 1.0 # Leave 1s per seq for safety
    if time_per_seq < 5.0:
        time_per_seq = 5.0
        
    # Pick top 3 candidates to optimize
    num_to_optimize = min(3, len(initial_chains))
    
    for c in initial_chains[:num_to_optimize]:
        if _time_left() < 2.0:
            break
            
        # Allocate remaining time for this sequence, split among candidates
        # Simple strategy: give each candidate 1/3 of the time allocated for this sequence
        alloc = (time_per_seq / num_to_optimize) * 0.8 # 20% buffer
        
        if alloc < 1.0:
            alloc = 1.0
            
        optimized = _local_search(sequence, c, alloc, rng)
        if optimized:
            e = _energy(sequence, optimized)
            if e < best_energy:
                best_energy = e
                best_chain = optimized
                
    # Fallback
    if best_chain is None:
        best_chain = initial_chains[0]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
