"""
HP Lattice Protein Folding Optimizer
Uses Simulated Annealing with Rigid Segment Relocation moves.
"""

import time
import random
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _neighbors(point):
    x, y = point
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def _calculate_energy(sequence, chain):
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for n in _neighbors(p):
                j = pos_to_idx.get(n)
                if j is not None and j > i + 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy

def _init_spiral(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [(0,0)]
        
    chain = [(0,0)]
    occupied = {(0,0)}
    x, y = 0, 0
    dx, dy = 1, 0
    dist = 1
    steps_in_current_dir = 0
    turns = 0
    
    for _ in range(n - 1):
        nx, ny = x + dx, y + dy
        if (nx, ny) in occupied:
            # Fallback
            return _init_random(sequence, dimension)
        
        x, y = nx, ny
        chain.append((x, y))
        occupied.add((x, y))
        steps_in_current_dir += 1
        
        if steps_in_current_dir == dist:
            steps_in_current_dir = 0
            # Turn Left
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                dist += 1
                
    return chain

def _init_random(sequence, dimension):
    n = len(sequence)
    chain = [(0,0)]
    occupied = {(0,0)}
    x, y = 0, 0
    rng = random.Random(42)
    
    for _ in range(n - 1):
        neighbors = _neighbors((x, y))
        random.shuffle(neighbors, rng)
        moved = False
        for nx, ny in neighbors:
            if (nx, ny) not in occupied:
                x, y = nx, ny
                chain.append((x, y))
                occupied.add((x, y))
                moved = True
                break
        if not moved:
            break
            
    if len(chain) < n:
        return [(i, 0) for i in range(n)]
        
    return chain

def _try_segment_move(chain, i, length, new_start, occupied):
    """
    Attempts a rigid segment move.
    Returns (new_chain, new_occupied_set) or None.
    """
    N = len(chain)
    
    # 1. Identify anchors
    left_anchor = chain[i-1] if i > 0 else None
    right_anchor = chain[i+length] if i + length < N else None
    
    # 2. Check left anchor connectivity
    if left_anchor is not None:
        if abs(new_start[0] - left_anchor[0]) + abs(new_start[1] - left_anchor[1]) != 1:
            return None
            
    # 3. Construct new segment using relative displacements
    # Get relative displacements of the original segment
    rel_disps = []
    for k in range(1, length):
        dx = chain[i+k][0] - chain[i+k-1][0]
        dy = chain[i+k][1] - chain[i+k-1][1]
        rel_disps.append((dx, dy))
        
    # Build new segment
    new_segment = [new_start]
    cx, cy = new_start
    for dx, dy in rel_disps:
        cx += dx
        cy += dy
        new_segment.append((cx, cy))
        
    # 4. Check right anchor connectivity
    if right_anchor is not None:
        last_new_pos = new_segment[-1]
        if abs(last_new_pos[0] - right_anchor[0]) + abs(last_new_pos[1] - right_anchor[1]) != 1:
            return None
            
    # 5. Check collisions
    # Create set of positions occupied by the rest of the chain
    rest_occupied = set()
    for idx, pos in enumerate(chain):
        if idx < i or idx >= i + length:
            rest_occupied.add(pos)
            
    # Check new segment against rest_occupied and itself
    seen = set()
    for pos in new_segment:
        if pos in seen:
            return None
        if pos in rest_occupied:
            return None
        seen.add(pos)
        
    # 6. Construct new chain and occupied set
    # We return a new occupied set to avoid in-place modification issues
    new_occupied = set(occupied)
    for pos in chain[i:i+length]:
        new_occupied.remove(pos)
    for pos in new_segment:
        new_occupied.add(pos)
        
    new_chain = chain[:i] + new_segment + chain[i+length:]
    
    return new_chain, new_occupied

def fold(sequence, dimension):
    _init_time()
    
    N = len(sequence)
    if N == 0:
        return []
        
    # Initial conformation
    best_chain = _init_spiral(sequence, dimension)
    best_energy = _calculate_energy(sequence, best_chain)
    
    # If spiral is bad, try random
    if best_energy == 0:
        rand_chain = _init_random(sequence, dimension)
        rand_energy = _calculate_energy(sequence, rand_chain)
        if rand_energy < best_energy:
            best_chain = rand_chain
            best_energy = rand_energy
            
    current_chain = best_chain
    current_energy = best_energy
    occupied = set(current_chain)
    
    rng = random.Random(12345)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    # Time budget
    # Total 115s for 10 sequences. ~11.5s per sequence.
    # We will use 11.0s to be safe.
    start_time = time.time()
    deadline = start_time + 11.0
    
    while time.time() < deadline:
        if T < min_T:
            # Restart
            current_chain = best_chain
            current_energy = best_energy
            occupied = set(best_chain)
            T = 1.0
            continue
            
        # Generate move
        # Segment length
        max_len = min(15, N - 2)
        if max_len < 2:
            break # Too short to move segments
            
        seg_len = rng.randint(2, max_len)
        
        # Start index
        # i can range from 0 to N - seg_len - 1
        i = rng.randint(0, N - seg_len - 1)
        
        # Candidate new starts
        candidates = []
        
        # Primary candidate source: Neighbors of the left anchor
        if i > 0:
            left_anchor = current_chain[i-1]
            for n in _neighbors(left_anchor):
                if n not in occupied:
                    candidates.append(n)
        else:
            # If i == 0, no left anchor. 
            # We can try neighbors of the current start.
            for n in _neighbors(current_chain[0]):
                if n not in occupied:
                    candidates.append(n)
                    
        # If no candidates from left anchor, try random empty spots in a local box?
        # Or just skip.
        if not candidates:
            T *= cooling_rate
            continue
            
        new_start = rng.choice(candidates)
        
        # Try move
        result = _try_segment_move(current_chain, i, seg_len, new_start, occupied)
        
        if result:
            new_chain, new_occupied = result
            new_energy = _calculate_energy(sequence, new_chain)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                # Accept
                current_chain = new_chain
                current_energy = new_energy
                occupied = new_occupied
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain
            # If rejected, occupied remains unchanged (we used a copy in _try_segment_move)
            
        T *= cooling_rate
        
    return [list(p) for p in best_chain]
