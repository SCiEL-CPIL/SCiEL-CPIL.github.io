import random
import math
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _initial_spiral(n):
    """
    Generate a compact spiral self-avoiding walk of length n.
    """
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    # Directions: East, North, West, South
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    current_dir = 0
    steps_in_dir = 1
    dir_change_count = 0
    
    x, y = 0, 0
    
    for i in range(1, n):
        for _ in range(steps_in_dir):
            if len(chain) >= n:
                return chain
            
            dx, dy = dirs[current_dir]
            nx, ny = x + dx, y + dy
            new_pos = (nx, ny)
            
            # Standard spiral is self-avoiding, so no check needed, 
            # but we add to occupied for safety if logic changes.
            x, y = nx, ny
            chain.append((x, y))
            occupied.add((x, y))
        
        # Change direction
        current_dir = (current_dir + 1) % 4
        dir_change_count += 1
        if dir_change_count % 2 == 0:
            steps_in_dir += 1
            
    return chain

def _calculate_energy(sequence, chain):
    """
    Calculate energy: - (number of H-H contacts not adjacent in sequence).
    """
    if not chain:
        return 0
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_positions = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.add(p)
    
    contacts = 0
    for p in h_positions:
        for q in _neighbours(p):
            if q in pos_to_idx:
                i = pos_to_idx[p]
                j = pos_to_idx[q]
                if q in h_positions:
                    if abs(i - j) > 1:
                        if i < j:
                            contacts += 1
    return -contacts

def _pivot_move(chain, k, angle):
    """
    Attempt a pivot move.
    Pivot at index k. Segment [k:] rotates around chain[k-1].
    """
    N = len(chain)
    if k <= 0 or k >= N - 1:
        return None
    
    pivot_point = chain[k-1]
    segment = chain[k:]
    
    new_segment = []
    for p in segment:
        dx = p[0] - pivot_point[0]
        dy = p[1] - pivot_point[1]
        
        if angle == 90:
            nx = -dy
            ny = dx
        elif angle == -90:
            nx = dy
            ny = -dx
        elif angle == 180:
            nx = -dx
            ny = -dy
        else:
            return None
            
        new_p = (pivot_point[0] + nx, pivot_point[1] + ny)
        new_segment.append(new_p)
        
    fixed_part = chain[:k]
    fixed_set = set(fixed_part)
    
    for p in new_segment:
        if p in fixed_set:
            return None
            
    new_chain = fixed_part + new_segment
    return new_chain

def _reversal_move(chain, k):
    """
    Reverse the segment chain[k:].
    """
    N = len(chain)
    if k <= 0 or k >= N:
        return None
        
    fixed_part = chain[:k]
    segment = chain[k:]
    new_segment = segment[::-1]
    
    p1 = fixed_part[-1]
    p2 = new_segment[0]
    if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
        return None
        
    fixed_set = set(fixed_part)
    for p in new_segment:
        if p in fixed_set:
            return None
            
    new_chain = fixed_part + new_segment
    return new_chain

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    start_time = time.time()
    # Assume 10 sequences total, 120s budget. 
    # Aim for 10s per sequence to be safe.
    deadline = start_time + 10.0
    
    best_chain = None
    best_energy = 0
    
    initial_chains = []
    
    c1 = _initial_spiral(N)
    if len(c1) == N:
        initial_chains.append(c1)
        
    c2 = [(-p[0], p[1]) for p in c1]
    if len(c2) == N:
        initial_chains.append(c2)
        
    c3 = [(p[0], -p[1]) for p in c1]
    if len(c3) == N:
        initial_chains.append(c3)
        
    c4 = [(-p[0], -p[1]) for p in c1]
    if len(c4) == N:
        initial_chains.append(c4)
        
    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(N)])
        
    rng = random.Random(12345 + len(sequence))
    
    for init_chain in initial_chains:
        if time.time() > deadline - 1.0:
            break
            
        current_chain = init_chain
        current_energy = _calculate_energy(sequence, current_chain)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = current_chain
            best_energy = current_energy
            
        T = 1.0
        T_min = 0.01
        alpha = 0.95
        
        # Adjust iterations based on N
        # Larger N needs fewer iterations to stay within time
        max_iter = 20000 // (N // 20 + 1)
        
        for i in range(max_iter):
            if time.time() > deadline - 0.5:
                break
                
            move_type = rng.random()
            new_chain = None
            
            if move_type < 0.7:
                k = rng.randint(1, N - 2)
                angle = rng.choice([90, -90, 180])
                new_chain = _pivot_move(current_chain, k, angle)
            else:
                k = rng.randint(1, N - 1)
                new_chain = _reversal_move(current_chain, k)
                
            if new_chain is None:
                continue
                
            new_energy = _calculate_energy(sequence, new_chain)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_chain = current_chain
                    best_energy = current_energy
                    
            T *= alpha
            if T < T_min:
                T = T_min
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(N)]
        
    return [list(p) for p in best_chain]
