"""
HP lattice protein folding using Simulated Annealing.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global time management
_START_TIME = time.time()
# The prompt says 120s CPU budget for all sequences, and 180s wall-clock for evaluation.
# The kill is at 210s. We set a conservative deadline to ensure we return a result.
# We will manage time per sequence allocation roughly, but a global guard is safer.
GLOBAL_DEADLINE = _START_TIME + 170.0  # 170s limit, leaving 40s margin for overhead/kill

def _neighbours(point):
    """Yield 4-neighbors of a point on a square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """
    Compute the energy of a conformation.
    Energy = - (number of H-H contacts that are not adjacent in the sequence).
    """
    if not chain:
        return 0
    
    # Map position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    
    # Iterate over all residues. 
    # Optimization: Only check H residues.
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        # Check 4 neighbors
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None:
                # Check if they are non-adjacent in the sequence
                if abs(i - j) > 1 and sequence[j] == 'H':
                    # To avoid double counting, only count if i < j
                    if i < j:
                        contacts += 1
                        
    return -contacts

def _initial_chain(sequence, rng):
    """
    Generate a random self-avoiding walk using a simple greedy/random growth.
    Returns None if it gets stuck (though for 2D lattice and these lengths, it's rare to fail completely if we restart, but possible to get trapped).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0)}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for nb in _neighbours(last):
            if nb not in occupied:
                # Prefer moves that don't create immediate dead ends? 
                # Simple random choice is fine for initialization.
                candidates.append(nb)
        
        if not candidates:
            return None
        
        # Choose randomly
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
        
    return chain

def _simulated_annealing(sequence, chain, rng, time_limit):
    """
    Perform Simulated Annealing on the given chain.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 1:
        return chain
    
    current_chain = list(chain)
    current_occupied = set(current_chain)
    current_energy = _compute_energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Cooling schedule
    # Start high enough to accept bad moves, end low.
    # Typical range for N=100 might be T=50 down to T=0.1
    start_temp = 30.0 
    end_temp = 0.01
    # We don't know the number of steps in advance, so we use time-based cooling.
    # We estimate steps per second.
    
    start_time = time.time()
    step_count = 0
    
    while time.time() < time_limit:
        step_count += 1
        # Dynamic temperature based on time elapsed in this SA run
        elapsed = time.time() - start_time
        # Assume we want to finish cooling in ~2 seconds per SA run? 
        # Or just use a fixed number of steps if time allows.
        # Let's use a fixed number of steps per SA run to keep it simple and predictable, 
        # but check time frequently.
        
        # Actually, a better strategy: Run for a fixed number of steps (e.g., 1000) 
        # or until time runs out. 
        # Let's just run steps until time_limit is reached.
        
        # Temperature decay: exponential
        # If we run ~1000 steps in 1s, and total time per seq is ~10s, 
        # we can just use a linear or exponential decay based on step count if we estimate total steps.
        # Simpler: T = start_temp * (end_temp/start_temp) ** (elapsed / total_estimated_time)
        # Let's assume total estimated time for this SA run is 2 seconds.
        T = start_temp * math.exp(-5.0 * elapsed) 
        if T < end_temp:
            T = end_temp
            
        # Generate a random move
        move_type = rng.random()
        
        new_chain = None
        new_occupied = None
        
        if move_type < 0.5:
            # Pivot move
            # Choose a pivot index i (1 to n-2)
            # Choose a rotation: 90, 180, 270
            # The segment from i to n-1 is rotated around chain[i-1]
            
            # Pick pivot
            i = rng.randint(1, n - 2)
            pivot_pos = current_chain[i - 1]
            
            # Extract segment
            segment = current_chain[i:]
            
            # Determine rotation
            rot = rng.choice([1, 2, 3]) # 1: 90 deg, 2: 180 deg, 3: 270 deg
            
            # Rotate points
            # Rotation around (px, py)
            # 90 deg CCW: (x,y) -> (px - (y-py), py + (x-px))
            # 180 deg: (x,y) -> (2*px - x, 2*py - y)
            # 270 deg CCW: (x,y) -> (px + (y-py), py - (x-px))
            
            rotated_segment = []
            for (x, y) in segment:
                if rot == 1:
                    nx = pivot_pos[0] - (y - pivot_pos[1])
                    ny = pivot_pos[1] + (x - pivot_pos[0])
                elif rot == 2:
                    nx = 2 * pivot_pos[0] - x
                    ny = 2 * pivot_pos[1] - y
                else: # rot == 3
                    nx = pivot_pos[0] + (y - pivot_pos[1])
                    ny = pivot_pos[1] - (x - pivot_pos[0])
                rotated_segment.append((nx, ny))
                
            # Construct new chain
            prefix = current_chain[:i]
            new_chain = prefix + rotated_segment
            
            # Check self-avoidance
            # We can check if any point in rotated_segment is in prefix (excluding the connection point which is valid)
            # Or simply check the whole new chain for duplicates.
            # For N=100, set construction is fast.
            new_occupied_set = set(new_chain)
            if len(new_occupied_set) != n:
                continue # Collision
            
            new_occupied = new_occupied_set
            
        elif move_type < 0.8:
            # Slither move (move the tail)
            # Pick a length k for the tail to move? 
            # Standard slither: move the last segment by 1 step.
            # Let's move the last k residues? 
            # Simpler: Move the last residue by 1 step.
            # Or move a small segment at the end.
            # Let's try moving the last k residues where k is small (1-5).
            k = rng.randint(1, 5)
            if k > n:
                k = n
            
            prefix = current_chain[:-k]
            tail = current_chain[-k:]
            
            # The first point of the new tail must be a neighbor of the last point of prefix.
            # The rest of the tail must maintain connectivity.
            # This is complex to generate randomly without collisions.
            # Alternative: "End move" - try all 4 neighbors of the last residue.
            # Let's stick to simple end moves for reliability.
            
            # Try moving the last residue
            last = current_chain[-1]
            prev = current_chain[-2] if n > 1 else None
            
            candidates = []
            for nb in _neighbours(last):
                if nb == prev:
                    continue
                if nb not in current_occupied:
                    candidates.append(nb)
            
            if not candidates:
                continue
            
            new_last = rng.choice(candidates)
            new_chain = current_chain[:-1] + [new_last]
            
            # Check self-avoidance (only last element changed, so just check if new_last is in prefix)
            # current_occupied contains all points. 
            # We removed 'last' and added 'new_last'.
            # If new_last was in current_occupied (and not 'last'), it's a collision.
            # Since we checked 'new_last not in current_occupied', and 'last' is in current_occupied, 
            # new_last is definitely not 'last' (unless it's the same point, which is excluded by 'nb not in occupied' logic? No, 'last' IS in occupied. So 'nb not in occupied' ensures nb != last).
            # So new_chain is valid SAW.
            
            new_occupied = (current_occupied - {last}) | {new_last}
            
        else:
            # Flip move (internal)
            # Pick an index i (1 to n-2)
            i = rng.randint(1, n - 2)
            pos = current_chain[i]
            prev = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            candidates = []
            for nb in _neighbours(pos):
                if nb == prev or nb == next_pos:
                    continue
                if nb not in current_occupied:
                    candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Check if new_pos collides with the rest of the chain
            # We removed pos, added new_pos.
            # If new_pos is in (current_occupied - {pos}), it's a collision.
            # We already checked 'nb not in current_occupied', so new_pos is not in current_occupied.
            # Thus, it's not in the rest of the chain.
            # So it is a valid SAW.
            
            new_chain = current_chain[:i] + [new_pos] + current_chain[i + 1:]
            new_occupied = (current_occupied - {pos}) | {new_pos}

        if new_chain is None:
            continue
            
        # Calculate new energy
        new_energy = _compute_energy(sequence, new_chain)
        
        delta_e = new_energy - current_energy
        
        # Acceptance probability
        if delta_e < 0:
            # Better, always accept
            accept = True
        else:
            # Worse, accept with probability exp(-delta_e / T)
            if T > 0:
                prob = math.exp(-delta_e / T)
            else:
                prob = 0.0
            accept = rng.random() < prob
            
        if accept:
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using multiple restarts of Simulated Annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Check global deadline
    if time.time() > GLOBAL_DEADLINE:
        # Fallback: return a straight line (valid SAW)
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain_overall = None
    best_energy_overall = 0 # Energy is negative, so 0 is "worse" than -1
    
    # Determine time budget for this sequence
    # We have ~170s total. 
    # If we assume 10 sequences, we have ~17s each.
    # But we don't know how many sequences are left.
    # Simple strategy: Use a fixed number of SA runs, or run until a local time limit.
    # Let's run SA for a fixed number of steps or time slice.
    
    # Let's do multiple independent restarts.
    # Each restart: Generate initial chain, run SA for ~1 second.
    
    max_restarts = 10
    time_per_restart = 1.5 # seconds
    
    for r in range(max_restarts):
        # Check global deadline
        if time.time() > GLOBAL_DEADLINE - 5.0: # 5s margin
            break
            
        # Check if we have enough time for this restart
        if time.time() + time_per_restart > GLOBAL_DEADLINE - 5.0:
            # Do a shorter run or skip?
            # Let's just run with a shorter limit
            current_limit = GLOBAL_DEADLINE - 5.0 - time.time()
            if current_limit < 0.1:
                break
        else:
            current_limit = time.time() + time_per_restart

        # Generate initial chain
        init_chain = _initial_chain(sequence, rng)
        if init_chain is None:
            # If greedy fails, try a simple straight line or skip
            # Straight line is always valid
            init_chain = [(i, 0) for i in range(n)]
        
        # Run SA
        sa_chain = _simulated_annealing(sequence, init_chain, rng, current_limit)
        
        # Evaluate
        energy = _compute_energy(sequence, sa_chain)
        
        if best_chain_overall is None or energy < best_energy_overall:
            best_chain_overall = sa_chain
            best_energy_overall = energy
            
    # If no valid chain found (shouldn't happen), return straight line
    if best_chain_overall is None:
        best_chain_overall = [(i, 0) for i in range(n)]
        
    # Convert to list of lists as required
    return [list(p) for p in best_chain_overall]

# EVOLVE-BLOCK-END
