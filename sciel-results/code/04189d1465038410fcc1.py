"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    # For efficiency, we can use a set for occupied positions
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n starting at (0,0).
    Uses a simple random walk with backtracking if stuck.
    If it fails to complete n steps, it returns None.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0,0)]
    
    chain = [(0,0)]
    occupied = {(0,0)}
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for _ in range(1, n):
        last = chain[-1]
        # Shuffle directions to ensure randomness
        rng.shuffle(directions)
        moved = False
        for d in directions:
            nx, ny = last[0] + d[0], last[1] + d[1]
            if (nx, ny) not in occupied:
                chain.append((nx, ny))
                occupied.add((nx, ny))
                moved = True
                break
        if not moved:
            # Stuck, backtrack
            if len(chain) <= 1:
                return None
            last_pos = chain.pop()
            occupied.remove(last_pos)
            # Try again from new last
            # To avoid infinite loops in pathological cases, we can limit backtracking depth
            # But for SAWs of length < 100, this usually resolves quickly.
            # A simple retry logic:
            for _retry in range(50):
                rng.shuffle(directions)
                moved = False
                for d in directions:
                    nx, ny = chain[-1][0] + d[0], chain[-1][1] + d[1]
                    if (nx, ny) not in occupied:
                        chain.append((nx, ny))
                        occupied.add((nx, ny))
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                return None
    return chain

def _pivot_move(chain, pivot_idx, new_dir, rng):
    """
    Apply a pivot move.
    pivot_idx: index in the chain where the pivot occurs (0 to n-2).
    The segment chain[pivot_idx+1:] is rotated by 90, 180, or 270 degrees 
    relative to the vector (chain[pivot_idx] -> chain[pivot_idx+1]).
    
    Returns None if the move results in a self-intersection.
    Returns the new chain if valid.
    """
    n = len(chain)
    if pivot_idx < 0 or pivot_idx >= n - 1:
        return None
    
    pivot = chain[pivot_idx]
    first_moved = chain[pivot_idx + 1]
    
    # Vector from pivot to first_moved
    dx = first_moved[0] - pivot[0]
    dy = first_moved[1] - pivot[1]
    
    # Determine rotation
    # new_dir is 0, 1, 2, 3 corresponding to 0, 90, 180, 270 degrees
    # We need to rotate the relative coordinates of the tail segment.
    
    # Get the tail segment relative to pivot
    tail = chain[pivot_idx+1:]
    rel_tail = [(p[0]-pivot[0], p[1]-pivot[1]) for p in tail]
    
    # Apply rotation to relative coordinates
    # Rotation matrix for 90 deg CCW: (x,y) -> (-y, x)
    # Rotation matrix for 180 deg: (x,y) -> (-x, -y)
    # Rotation matrix for 270 deg CCW: (x,y) -> (y, -x)
    
    rotated_rel_tail = []
    for (x, y) in rel_tail:
        if new_dir == 0:
            nx, ny = x, y
        elif new_dir == 1: # 90 deg CCW
            nx, ny = -y, x
        elif new_dir == 2: # 180 deg
            nx, ny = -x, -y
        else: # 270 deg CCW
            nx, ny = y, -x
        rotated_rel_tail.append((nx, ny))
        
    # Reconstruct absolute coordinates
    new_tail = [(pivot[0] + x, pivot[1] + y) for (x, y) in rotated_rel_tail]
    
    # Check self-avoidance
    # The head part (0..pivot_idx) is unchanged.
    # The tail part (pivot_idx+1..n-1) is new_tail.
    # We need to check:
    # 1. No self-intersection within new_tail (should be fine if original was SAW and rotation is rigid, 
    #    but relative positions are preserved, so internal collisions are impossible if original had none).
    # 2. No intersection between new_tail and head_part (excluding the pivot which is shared).
    # 3. The connection between pivot and new_tail[0] is valid (unit step).
    
    # Check connection
    if new_tail:
        f0 = new_tail[0]
        dist = abs(f0[0]-pivot[0]) + abs(f0[1]-pivot[1])
        if dist != 1:
            return None
            
    # Check intersection with head
    head_set = set(chain[:pivot_idx+1])
    for p in new_tail:
        if p in head_set:
            return None
            
    # Construct new chain
    new_chain = chain[:pivot_idx+1] + new_tail
    
    return new_chain

def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Improve conformation via single-residue moves (kink moves) that preserve connectivity.
    This is a fast local search to polish the solution.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break
        
        i = rng.randint(1, n - 2)
        p = chain[i]
        tried = False

        # Try all 4 neighbours
        neighbours_list = list(_neighbours(p))
        rng.shuffle(neighbours_list)
        
        for np in neighbours_list:
            if np == p:
                continue
            if np in occupied and occupied[np] != i:
                continue
            
            # Check if moving to np preserves connectivity
            prev, nxt = chain[i-1], chain[i+1]
            dist_prev = abs(np[0]-prev[0]) + abs(np[1]-prev[1])
            dist_next = abs(np[0]-nxt[0]) + abs(np[1]-nxt[1])
            
            if dist_prev != 1 or dist_next != 1:
                continue
            
            # Perform move
            del occupied[p]
            chain[i] = np
            occupied[np] = i
            new_energy = _compute_energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                # Revert
                del occupied[np]
                chain[i] = p
                occupied[p] = i
                no_improve_count += 1
            
            tried = True
            break
        
        if not tried:
            no_improve_count += 1
        
        if no_improve_count > 200:
            break

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using Stochastic Local Search with Pivot Moves.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n <= 1:
        return [[0, 0]] * n

    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate initial random SAW
    # Try multiple times to get a valid SAW
    initial_chain = None
    for _ in range(100):
        initial_chain = _generate_random_saw(n, rng)
        if initial_chain:
            break
            
    if initial_chain is None:
        # Fallback: straight line (might not be optimal but valid)
        initial_chain = [(i, 0) for i in range(n)]
        
    current_chain = initial_chain
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Phase 2: SLS with Pivot Moves
    # We will run until deadline or no improvement for a long time
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < _DEADLINE - 10:
        # Choose a pivot point
        # Bias towards smaller pivots for local changes, but allow large ones
        # Uniform is a good start
        pivot_idx = rng.randint(0, n - 2)
        new_dir = rng.randint(1, 3) # 1, 2, 3 rotations (0 is identity)
        
        new_chain = _pivot_move(current_chain, pivot_idx, new_dir, rng)
        
        if new_chain is None:
            no_improve += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Acceptance probability for uphill moves (Simulated Annealing style)
            # T decreases over time
            # Simple geometric cooling
            # We don't track T explicitly, just use a fixed probability that decays with no_improve
            # Or use a simple SA schedule
            # Let's use a simple SA: T = T0 * (cooling_rate)^step
            # But we don't have a global step counter easily accessible in this loop structure without overhead.
            # Let's just use a fixed small probability for uphill moves to escape local minima
            # P = exp(-delta / T)
            # Let T be dynamic based on no_improve?
            # Simpler: if no_improve is high, increase exploration.
            # Let's use a fixed temperature that decays slowly.
            
            # Calculate delta
            delta = current_energy - new_energy # positive if uphill
            if delta > 0:
                # Accept with probability exp(-delta / T)
                # T starts at 5.0 and cools
                # We can approximate T based on iteration count
                # Let's just use a simple rule: accept uphill moves with prob 0.1 if delta is small
                # Or use a proper SA.
                # Let's implement a simple SA schedule inside the loop.
                
                # We'll maintain a temperature variable
                # Since we can't easily stateful across loop iterations without extra vars, 
                # let's just use a fixed acceptance rate for now, or re-implement with T.
                
                # Re-doing with T:
                # Actually, let's just use a fixed probability for now to keep it simple and robust.
                # If we accept too many bad moves, we waste time.
                # If we accept too few, we get stuck.
                # Let's try: accept if rng.random() < 0.05
                if rng.random() < 0.05:
                    current_chain = new_chain
                    current_energy = new_energy
                    no_improve += 1 # It's an uphill move, so it didn't improve best
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        if no_improve > max_no_improve:
            break
            
    # Phase 3: Local Improvement (Kink Moves)
    # Use remaining time for local polishing
    if time.time() < _DEADLINE - 2:
        improved_chain, improved_energy = _local_improve(sequence, best_chain, rng, max_moves=5000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
