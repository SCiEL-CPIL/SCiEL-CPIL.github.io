# EVOLVE-BLOCK-START

import random
import time
import math

def _manhattan(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2])

def _generate_initial_chain(seq, dim, rng):
    """Generate a random self-avoiding walk."""
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    chain = [(0,0,0)]
    occupied = {(0,0,0): 0}
    
    # Directions: x, y, z +/-
    dirs = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
    ]
    
    for i in range(1, n):
        valid_moves = []
        last = chain[-1]
        # Get previous direction to avoid immediate backtracking (optional, but helps)
        if i > 1:
            prev_dir = (last[0]-chain[-2][0], last[1]-chain[-2][1], last[2]-chain[-2][2])
        else:
            prev_dir = None
            
        for d in dirs:
            # Avoid immediate reversal
            if prev_dir and d == (-prev_dir[0], -prev_dir[1], -prev_dir[2]):
                continue
                
            new_pos = (last[0]+d[0], last[1]+d[1], last[2]+d[2])
            if new_pos not in occupied:
                valid_moves.append((d, new_pos))
                
        if not valid_moves:
            # Trapped, restart or fail? For now, return None to signal failure
            return None
            
        d, new_pos = rng.choice(valid_moves)
        chain.append(new_pos)
        occupied[new_pos] = i
        
    return chain

def _calc_energy_fast(seq, chain, occupied):
    """Calculate energy using occupied dict."""
    n = len(chain)
    hh_contacts = 0
    h_indices = [i for i, c in enumerate(seq) if c == 'H']
    
    for i in h_indices:
        p = chain[i]
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            q = (p[0]+dx, p[1]+dy, p[2]+dz)
            if q in occupied:
                j = occupied[q]
                if abs(i - j) > 1:
                    if i < j:
                        hh_contacts += 1
    return -hh_contacts

def _simulate_annealing(seq, dim, time_limit, rng):
    """
    Perform Simulated Annealing using Pivot Moves (Crankshaft).
    """
    n = len(seq)
    if n < 4:
        return None, 0
        
    # 1. Generate initial chain
    chain = None
    for _ in range(50):
        chain = _generate_initial_chain(seq, dim, rng)
        if chain:
            break
            
    if not chain:
        return None, 0
        
    # Track occupied positions for fast lookup
    occupied = {pos: i for i, pos in enumerate(chain)}
    current_energy = _calc_energy_fast(seq, chain, occupied)
    best_energy = current_energy
    best_chain = list(chain)
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    start_time = time.time()
    
    # Estimate steps per second to adjust cooling rate? 
    # Or just fix a number of iterations and adjust T based on time.
    # Let's do a fixed number of iterations and adjust T to fit time limit?
    # Better: Run until time limit.
    
    # We will use a geometric cooling schedule but adapt the step size to time.
    # Let's assume we can do ~10000 moves per second.
    # Total moves = time_limit * 10000
    # T = T_start * (T_end/T_start)^(step/total_steps)
    
    # To be safe, let's just use a simple cooling factor per move.
    # If we do 50000 moves, factor = (0.01/1.0)^(1/50000) ~ 0.99989
    # Let's dynamically calculate based on a target number of moves.
    target_moves = int(time_limit * 5000) # 5000 moves/sec estimate
    if target_moves < 100:
        target_moves = 100
        
    cooling_rate = (T_end / T_start) ** (1.0 / target_moves)
    T = T_start
    
    move_count = 0
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Choose a pivot move
        # Pivot at i, rotate segment [i+1, j]
        # i in [1, n-2]
        # j in [i+1, n-1]
        # Segment length L = j - i
        
        # Random pivot
        i = rng.randint(1, n - 2)
        
        # Random segment length, biased towards short segments?
        # Max length is n - 1 - i
        max_len = n - 1 - i
        if max_len < 1:
            continue
        # Sample length with preference for shorter?
        # Uniform is fine, or geometric. Let's use uniform for simplicity.
        L = rng.randint(1, max_len)
        j = i + L
        
        # Get segment positions
        # segment = chain[i+1 : j+1]
        # pivot_pos = chain[i]
        # prev_pos = chain[i-1]
        
        # To rotate, we need to find a new direction for the segment start.
        # The segment starts at chain[i+1] and ends at chain[j].
        # The vector from chain[i] to chain[i+1] is the initial direction.
        # We rotate this vector around the pivot axis?
        # Actually, standard crank move:
        # The segment [i+1, j] is rigid.
        # We move the whole segment.
        # The new position of the segment start (i+1) must be adjacent to pivot (i).
        # And the relative coordinates of the rest of the segment remain the same.
        
        # Let's define the relative coordinates of the segment with respect to chain[i+1]
        # Or better, with respect to the pivot chain[i].
        # Let v_k = chain[i+k] - chain[i] for k in 1..L
        # We need to rotate the vector v_1 to a new direction d_new which is a neighbor of origin (0,0,0)
        # such that the new positions don't collide.
        
        # Current direction d_old = chain[i+1] - chain[i]
        d_old = (chain[i+1][0]-chain[i][0], chain[i+1][1]-chain[i][1], chain[i+1][2]-chain[i][2])
        
        # Possible new directions: all 6 neighbors of origin, excluding d_old and -d_old?
        # Actually, we can rotate to any neighbor except the one we came from?
        # No, we can rotate to any neighbor that is not d_old (since that's the current pos).
        # We must also ensure the new segment doesn't collide with the rest of the chain.
        
        # Let's try all 6 directions for the new start of the segment.
        dirs = [
            (1,0,0), (-1,0,0),
            (0,1,0), (0,-1,0),
            (0,0,1), (0,0,-1)
        ]
        
        # Filter out d_old
        possible_dirs = [d for d in dirs if d != d_old]
        
        if not possible_dirs:
            continue
            
        # Shuffle to randomize
        rng.shuffle(possible_dirs)
        
        found_move = False
        for d_new in possible_dirs:
            # Calculate new positions
            # new_pos[k] = chain[i] + d_new * k + relative_offset?
            # No, the segment is rigid.
            # The transformation is a rotation of the entire segment vector.
            # It's easier to just translate? No, pivot move rotates around i.
            # So chain[i+k]_new = chain[i] + R * (chain[i+k] - chain[i])
            # Where R is the rotation matrix corresponding to d_old -> d_new.
            
            # How to find R?
            # d_old is a unit vector. d_new is a unit vector.
            # There are multiple rotations.
            # For simplicity in lattice, we can just check if the new segment fits.
            # Actually, for a 90 degree rotation around an axis, the relative coordinates change.
            
            # Let's use a simpler approach:
            # The segment [i+1, j] moves such that chain[i+1] goes to chain[i] + d_new.
            # The rest of the segment [i+2, j] maintains its relative shape to chain[i+1].
            # So, new_chain[i+1] = chain[i] + d_new
            # new_chain[i+k] = new_chain[i+1] + (chain[i+k] - chain[i+1])
            
            # This is a "translation" of the segment relative to the new start,
            # but the start itself has moved relative to the pivot.
            # This is effectively a crank move.
            
            # Check for collisions
            # The segment occupies positions new_chain[i+1] ... new_chain[j]
            # We need to check if any of these positions are occupied by chain[0..i-1] or chain[j+1..n-1]
            
            # Pre-compute relative offsets
            # rel[k] = chain[i+k] - chain[i+1] for k=0..L-1 (where k=0 is the start of segment)
            # rel[0] = (0,0,0)
            # rel[1] = chain[i+2] - chain[i+1]
            # ...
            
            # New start:
            new_start = (chain[i][0] + d_new[0], chain[i][1] + d_new[1], chain[i][2] + d_new[2])
            
            # Check if new_start is occupied (by 0..i-1 or j+1..n-1)
            # Note: chain[i] is the pivot, it's not "occupied" in the sense of being part of the moving segment,
            # but it IS occupied by the pivot itself. The segment moves INTO space adjacent to pivot.
            # So new_start must NOT be occupied by any index < i or > j.
            # It IS allowed to be occupied by i? No, i is the pivot, the segment starts at i+1.
            # So new_start is a new position.
            
            if new_start in occupied:
                # Check if it's occupied by the segment itself (impossible, we are moving)
                # or by pivot (impossible, pivot is chain[i], new_start is neighbor)
                # or by rest of chain.
                # If occupied by rest of chain, collision.
                # If occupied by i? No.
                # So if new_start in occupied, it must be from 0..i-1 or j+1..n-1.
                # Wait, what if new_start is chain[i+1]? That's d_old, which we excluded.
                # So if new_start in occupied, it's a collision.
                continue
                
            # Check rest of segment
            valid = True
            new_positions = [new_start]
            for k in range(2, L + 1):
                # new_chain[i+k] = new_chain[i+1] + (chain[i+k] - chain[i+1])
                rel = (chain[i+k][0] - chain[i+1][0], 
                       chain[i+k][1] - chain[i+1][1], 
                       chain[i+k][2] - chain[i+1][2])
                pos = (new_start[0] + rel[0], new_start[1] + rel[1], new_start[2] + rel[2])
                new_positions.append(pos)
                
                if pos in occupied:
                    # Collision with rest of chain
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check if new_positions collide with each other?
            # The segment was self-avoiding, and we just translated/rotated it rigidly?
            # Yes, relative distances are preserved. So no internal collisions.
            
            # Okay, move is valid.
            # Update energy
            # Remove old contacts for indices i+1..j
            # Add new contacts for indices i+1..j
            
            # We need to update occupied dict
            # And active_hh set
            
            # List of old positions
            old_positions = chain[i+1 : j+1]
            
            # Apply move
            for k in range(i+1, j+1):
                occupied.pop(chain[k])
                chain[k] = new_positions[k - (i+1)]
                occupied[chain[k]] = k
                
            new_energy = _calc_energy_fast(seq, chain, occupied)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                for k in range(i+1, j+1):
                    occupied.pop(chain[k])
                    chain[k] = old_positions[k - (i+1)]
                    occupied[chain[k]] = k
            
            found_move = True
            break # Only try one direction per iteration? Or all?
            # Trying all valid directions and picking the best or random one is better.
            # But for SA, we usually propose one move.
            # Let's stick to one proposed move per iteration to keep T schedule simple.
            
        # End for d_new
        
        move_count += 1
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n < 4:
        return [list((i, 0, 0)) for i in range(n)]
        
    # Time budget
    # Total 240s for 26 sequences.
    # ~9s per sequence.
    # We will use 8s for SA.
    time_budget = 8.0
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Multiple restarts
    # If n is small, more restarts. If large, fewer.
    restarts = 3
    
    for r in range(restarts):
        t_limit = time_budget / restarts
        chain, energy = _simulate_annealing(sequence, dimension, t_limit, rng)
        if chain and energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback
        best_chain = [list((i, 0, 0)) for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END
